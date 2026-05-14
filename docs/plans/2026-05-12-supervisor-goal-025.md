# Supervisor Goal 025 — W3.3 RCA-5 + Launch-Amortization Fix (Combined)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G24 V3 scale sweep APPROVED structurally (all M24 metrics green except D7a outcome). Commit `429c2cca` on `bench-spike/w33-slice-aware-scale-validation`. superhub-50K under V3 + slice-aware: baseline 1248.361 µs, merge-resident 2249.337 µs, paired delta **+1000.976 µs / +80.18% / ratio 0.555×** — slice-aware is **1.8× SLOWER** at this fixture size. Root cause: `grid_dim.x = slice_count = 468` (4× more than baseline's 117) × ~2.5 µs/launch ≈ +750-1000 µs overhead consumed all work-balancing savings. G23 design's physics works (stddev −49.85% real) but amortization is wrong.
**Date:** 2026-05-12.

---

## Context

User directive recorded 2026-05-12: *"NO ANY FUCKING DEFERS, TOYSHIT, SIMPLIFICATION. SETUP A CLEAR GOAL TO FIX THE ROOT CAUSE AND DELIVER FULL AND ROBUST IMPLEMENTATION!!!!"* (carry-forward from G22 context, reaffirmed by G25 option selection: "RCA-5 + direct fix combined").

G24's verdict is the first **design-level performance issue** discovered in the chain (vs prior measurement-noise findings or scaffolding-stub findings). G25 is a combined RCA + fix because the RCA scope is narrow (attribute the +1001 µs across known launch-overhead buckets) and the fix is well-understood (CUDA grid-stride loop pattern). Splitting into two goals adds dispatch overhead with no diagnostic benefit.

### What we already know going into G25

1. **The physics is right.** G23 produces measurably better per-block work distribution (stddev 459.998 → 230.670 = −49.85% on superhub-50K).
2. **Row-equality holds.** Output is bit-identical to baseline (29,539 triples both paths).
3. **The wall-clock cost is wrong.** Paired delta +1001 µs / +80% slowdown.
4. **The proximate cause is the launch count.** `grid_dim.x = slice_count = 468` for the skewed path vs baseline `ceil(n_xy/256) = 117` for the uniform path.
5. **The standard CUDA idiom for this is grid-stride loop.** Keep grid_dim at the baseline size; have each block iterate over multiple work units via an outer loop.

### The fix in code-pattern terms

Current (G23) `wcoj_triangle_{count,materialize}_sliced` body:
```cuda
int start = slice_starts[blockIdx.x];
int end = slice_ends[blockIdx.x];
for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
    // body (count or materialize logic)
}
```

Target (G25) body — grid-stride over slices:
```cuda
for (int s = blockIdx.x; s < slice_count; s += gridDim.x) {
    int start = slice_starts[s];
    int end = slice_ends[s];
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        // body unchanged
    }
}
```

Combined with launch surface change: `grid_dim.x = ceil(n_xy / 256)` (or analogous baseline-equivalent computation), NOT `slice_count`. Each block now processes `slice_count / gridDim.x ≈ 4` slices on average.

### Expected effect on D7a measurement

If the +1001 µs paired delta is dominated by launch overhead (estimated +750 µs from 4× launches × 2-3 µs each + +250 µs from refresh-pipeline + parameter-marshaling overhead), reducing launches by 4× should drop the paired delta to roughly +250 µs / +20% / ratio ~0.83×. That's still slower than baseline at this fixture, NOT yet ≥ 2.0×. **The 50K fixture may be too small for work-balancing benefits to overcome ANY non-zero launch overhead.**

If RCA Phase A shows the launch-count is the only significant overhead source, then 50K is genuinely launch-overhead-limited and the design only shines at larger scales (200K+ where each block does enough work to amortize launch cost). G25 Phase B fix + spot-check 200K to verify scale-emergence.

---

## G25 — RCA-5 + launch-amortization fix combined

### Goal

Cut `feat/w33-slice-aware-launch-amortized` from `feat/w33-slice-aware-implementation @ dcb556db` (preserves G23's slice-aware production implementation; G24 had no impl changes so cutting from G23 keeps lineage clean). Two-phase work in one goal:

* **PHASE A (RCA-5, ~30 min):** Extend feature-gated probes to attribute the G24-measured +1001 µs paired delta across launch-overhead buckets. Output Table A.
* **PHASE B (FIX, ~60 min):** Modify slice-aware kernels + launch surface to amortize launches via grid-stride loop pattern. Re-measure at superhub-50K AND superhub-200K (spot-check) under V3 protocol. Output Table B.

Branch stays unmerged. Final commit must include the G25 evidence README with both Tables A and B + D7a verdict at both scales + recommendation for G26.

### Strategies (GQM+Strategies)

* **S25.1** Cut `feat/w33-slice-aware-launch-amortized` from `feat/w33-slice-aware-implementation @ dcb556db`. Worktree at `.worktrees/w33-launch-amortized`.

* **S25.2 — PHASE A: RCA-5 launch-overhead attribution.** Extend the existing feature-gated probes (`wcoj-phase-timing` gate, established in G16/G20/G22) with FOUR new buckets:
  * **`kernel_launch_driver`** — time spent in CUDA driver launch dispatch for each count + materialize launch. Wrap each kernel-launch call site in `provider/wcoj.rs` with `Instant::now()` brackets, separate from kernel execution time.
  * **`merge_refresh_pipeline`** — time spent in the Merge-refresh path: column download + weight compute + prefix-sum partition + device upload. Wrap the entire refresh body in `memory.rs:1170` with one bracket.
  * **`parameter_marshaling_delta`** — time spent preparing the `slice_starts` + `slice_ends` + `slice_count` parameters before each launch, vs the baseline parameter prep. Bracketed separately.
  * **`per_launch_sync`** — any `cudaDeviceSynchronize` or stream-sync cost induced by the slice-aware path that doesn't exist in baseline. Bracketed at each sync point.
  * All probes feature-gated under `wcoj-phase-timing` (zero overhead when off). Diff strictly within `cfg(feature)` blocks.

* **S25.3 — PHASE A measurement.** Run the existing `wcoj_design_behavior_probe` binary on superhub-50K once under the slice-aware path (G23's existing implementation) with the new probes enabled. Capture per-bucket µs to `docs/evidence/2026-05-12-w33-launch-amortized/phase_a_overhead_buckets.csv`. Sum the buckets and reconcile against G24's +1001 µs Criterion delta within reasonable noise band (±50 µs).

* **S25.4 — PHASE A verdict.** Build Table A: per-bucket µs + share of +1001 µs. Identify the dominant overhead source(s). State explicit attribution: "launch-overhead dominates" / "refresh-pipeline dominates" / "hybrid".

* **S25.5 — PHASE B fix application.** Modify `crates/xlog-cuda/kernels/wcoj.cu` to add grid-stride-over-slices loop to `wcoj_triangle_count_sliced` and `wcoj_triangle_materialize_sliced`:
  ```cuda
  for (int s = blockIdx.x; s < slice_count; s += gridDim.x) {
      int start = slice_starts[s];
      int end = slice_ends[s];
      for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
          // body identical to G23's
      }
  }
  ```
  Modify `crates/xlog-cuda/src/provider/wcoj.rs` to set `grid_dim.x = ceil(n_xy / 256)` (or analogous baseline-equivalent) for both sliced launches at provider/wcoj.rs:970 (count) and :1211 (materialize). NOT `slice_count`. The slice plan still passes through as device pointer + scalar.

* **S25.6 — PHASE B re-validation.** Re-run the G23 design-behavior probe on superhub-50K to confirm:
  * Row-equality PASS.
  * Per-block-output stddev STILL reduced vs baseline 459.998 (work-balancing preserved).
  * Grid blocks now match baseline ~117 (NOT 468).

* **S25.7 — PHASE B Criterion measurement.** Run the V3 Criterion scale sweep at superhub-50K AND superhub-200K under the fixed implementation. Capture paired delta + 95% CI + speedup ratio at both scales. Compute D7a verdict at each scale. (Skip 1M for time budget; if 50K shows ≥ 1.5× and 200K shows ≥ 2.0×, G26 closure runs the full sweep.)

* **S25.8 — PHASE B verdict.** Build Table B: per-scale paired delta + speedup ratio + D7a verdict + comparison vs G24 (1.8× slower) + comparison vs G21 (0.982× scaffolding-stub). State: "D7a clears at 50K" / "D7a clears at 200K only" / "D7a still fails at all measured scales".

* **S25.9 — D7b spot-check.** Re-run uniform-u32-10K under V3 protocol after the kernel-body change. Adaptive skew detection should keep this on the BASELINE kernel path (not slice-aware), so D7b should be effectively unchanged from G23's +1.347%. Verify: paired delta within ±5% budget.

* **S25.10 — All existing test gates green:**
  * `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run` EXIT 0.
  * `cargo build --release --features wcoj-phase-timing` EXIT 0.
  * `cargo build --release` (no feature) EXIT 0.
  * `cargo fmt --check --all` EXIT 0.

* **S25.11** Branch UNMERGED to all 15 parents (main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed, G19-scale-sweep, G20-stability-rca3, G21-scale-sweep-v3-stable, G22-design-behavior-rca4, G23-slice-aware-implementation, G24-scale-validation). No FF-merge, no push, no tag.

* **S25.12** Multi-commit allowed (Phase A + Phase B + evidence). Final commit must be the evidence README. Suggested:
  1. `feat(w33): RCA-5 launch-overhead probes (Phase A)`
  2. `feat(w33): grid-stride loop in sliced kernels + baseline grid_dim (Phase B fix)`
  3. `feat(w33): G25 evidence run (Tables A+B + D7a at 50K+200K)`

### Questions

* **Q25.1** Branch HEAD SHA?

**Phase A (RCA-5):**
* **Q25.2** Table A: per-bucket µs + share of G24 +1001 µs. Which bucket dominates?
* **Q25.3** Reconciliation sum vs +1001 µs: arithmetic + noise band.

**Phase B (Fix):**
* **Q25.4** Kernel modification: grid-stride loop applied to both `_sliced` kernels at wcoj.cu file:line?
* **Q25.5** Launch surface: `grid_dim.x = ceil(n_xy/256)` (or analogous) confirmed at provider/wcoj.rs file:line?
* **Q25.6** Row-equality PASS on superhub-50K + per-block stddev still reduced vs baseline?
* **Q25.7** Grid block count post-fix: ~117 (baseline) NOT 468?
* **Q25.8** superhub-50K under V3 fixed: paired delta + 95% CI + speedup ratio + D7a verdict?
* **Q25.9** superhub-200K under V3 fixed: same fields + D7a verdict?
* **Q25.10** D7b spot-check uniform-u32-10K paired delta + verdict (should be unchanged from G23 since uniform path bypasses _sliced kernel)?

**Test preservation:**
* **Q25.11** All existing tests green (workspace + CUDA cert 1/1)?

**Discipline:**
* **Q25.12** Branch unmerged from all 15 parents?
* **Q25.13** ZERO R6 anti-patterns introduced?
* **Q25.14** Both feature builds compile?

### Metrics

* **M25.1** `feat/w33-slice-aware-launch-amortized` exists; HEAD reachable from none of 15 parents.
* **M25.2** `docs/evidence/2026-05-12-w33-launch-amortized/README.md` exists with Tables A + B.
* **M25.3** Phase-A probe CSV `phase_a_overhead_buckets.csv` exists.
* **M25.4** Table A sum reconciles to +1001 µs ± 50 µs band; arithmetic in README.
* **M25.5** Table B per-scale paired delta + 95% CI + speedup ratio + D7a verdict at 50K + 200K.
* **M25.6** Row-equality PASS at 50K + 200K under fixed implementation.
* **M25.7** Per-block-output stddev under fixed implementation STILL strictly less than baseline 459.998.
* **M25.8** Grid block count post-fix on superhub-50K = baseline range (~117), NOT 468.
* **M25.9** D7b uniform-u32-10K paired delta ≤ ±5% under fixed implementation.
* **M25.10** All existing tests EXIT 0; CUDA cert 1/1.
* **M25.11** `cargo build --release --features wcoj-phase-timing` EXIT 0.
* **M25.12** `cargo build --release` (no feature) EXIT 0.
* **M25.13** `cargo fmt --check --all` EXIT 0.
* **M25.14** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "feat/w33*"` empty.
* **M25.15** Branch unmerged from all 15 parents.
* **M25.16** Zero R6 anti-patterns introduced (`git grep classify_heavy_rows` + `git grep mask_histogram` in W3.3 production code empty).

### Supervisor validation per locked protocol

* Read evidence README (Tables A + B) end-to-end.
* `git rev-parse feat/w33-slice-aware-launch-amortized` ≠ all 15 parent SHAs.
* Spot-check Table A reconciliation arithmetic.
* Verify M25.7 (stddev still reduced) and M25.8 (grid blocks back to baseline) — these are the "fix worked at the physics level" gates.
* Verify D7a verdict at 50K + 200K.
* Run both feature builds + fmt + CUDA cert from supervisor session.
* Verify branch unmerged + no tag + no origin push.

**Decision tree based on G25 verdict:**

* **D7a ≥ 2.0× at superhub-50K:** W3.3 essentially closes. G26 = closure proposal grounded in G11-G25 evidence chain, optional 1M cell run for completeness.
* **D7a < 2.0× at 50K but ≥ 2.0× at 200K:** Scale-threshold closure. G26 = closure proposal documenting scale threshold "≥ 200K".
* **D7a < 2.0× at both 50K and 200K but trending positive (ratio increases with scale):** G26 = 1M cell run; if ≥ 2.0×, close with 1M threshold; otherwise user-decision-required (defer? amend? continue tuning?).
* **D7a ratio doesn't improve under the fix OR row-equality breaks:** G26 = forensic on what went wrong with the grid-stride implementation.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `feat/w33-slice-aware-launch-amortized` into main in this goal.
* No `docs/v065-closure-board.md` edit (G26's conditional job).
* No `v0.6.6` references.
* **No new R6 anti-patterns:** no per-call histogram launch, no heavy/light kernel split, no per-call classify_heavy_rows kernel, no front-end mask_histogram/classify/partition_scan pass.
* No modification of existing `wcoj_triangle_count` or `wcoj_triangle_materialize` kernels (the baseline uniform path). Modify only the `_sliced` variants. Baseline must remain bit-identical for non-merge-resident invocations.
* No modification of the existing G23 slice-bin computation in memory.rs:1170 — the heuristic is correct (it's producing real work-balancing per stddev evidence); only the kernel launch/iteration pattern needs to change.
* No removal of row-equality assertions.
* No D7 amendment.
* No closure proposal in this goal.
* No "simplification" that abandons the slice-aware path entirely (e.g., always taking baseline). The fix MUST preserve work-balancing while reducing launch cost.
* No methodology change in the Criterion bench (V3 sample_size(200) + iters=1 + paired-batching stays unchanged).

### Why this goal is the chain's actual-closure moment

24 supervisor goals have produced: paper-aligned plan, root-cause attribution, slice-aware implementation, scale measurement, launch-overhead diagnosis. G25 either delivers the closure-ready performance result OR produces precise data for the final user-decision. The fix is well-understood (grid-stride loop is textbook CUDA), the RCA is narrow (4 named buckets), the validation is direct (V3 Criterion at 50K + 200K).

If G25 lands clean with D7a ≥ 2.0× at any scale, W3.3 closes via G26. If not, the chain produces empirically-grounded evidence for the user's final call between gate amendment / scale extension / fundamental redesign.

Proceed: cut `feat/w33-slice-aware-launch-amortized` from `dcb556db`, execute Phase A (probes + measurement + Table A), execute Phase B (grid-stride kernels + baseline grid_dim + re-validation), capture Tables A+B in evidence README, single bundled or multi-commit (final = README), emit REVIEW REQUEST with HEAD SHA + Table A dominant bucket + Table B per-scale D7a verdicts + recommendation for G26.
