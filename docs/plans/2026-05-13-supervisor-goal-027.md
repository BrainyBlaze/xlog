# Supervisor Goal 027 — W3.3 Persistent-Threads Work-Stealing + Heavy-Slice Splitting Redesign

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G26 NOT-CLOSURE-READY at commit `2aeb74b4` on `feat/w33-grid-amortized-v2`. **Two empirical findings forced the redesign:**
1. **Work-balancing physics is in fine-grained per-block slicing.** Variance reduction collapsed from G23's 49.85% (468 blocks × 1 slice each) to G26 Phase 2's 0.030% (117 blocks × ~4 chunked slices each). The "heavy block" max-5041-row slice is the wall-clock bottleneck regardless of slice metadata.
2. **D7b regression** at uniform-u32-10K (+40.63% paired delta) reveals Phase 2's launch-surface changes inadvertently broke the G23 adaptive skew detection that routed uniform inputs through the baseline kernel.
**Date:** 2026-05-13.

---

## Context

User directive (verbatim, 2026-05-12 carry-forward): *"NO TOY SHIT! NO SIMPLIFICATIONS! NO WORKAROUND! NO SILENT BYPASSES! ONLY PRODUCTION GRADE CODE! ONLY DEEP ROOT CAUSES INVESTIGATIONS! ONLY PROPER AND HIGHEST QUALITY ALGORITHMIC AND STRUCTURAL FIXES!"*

Meta-goal: deliver FULL v0.6.5 per `docs/v065-closure-board.md` + `ROADMAP.md`. W3.3 remains OPEN after 16 iteration branches. G27 is the **first structural redesign** — abandoning the static-grid approach (G23 468-blocks; G26 117-blocks) in favor of **dynamic work-stealing** via persistent threads.

### The architectural diagnosis from G26 data

| Variant | grid_dim | Slices | Variance reduction | D7a wall-time | Verdict |
|---|---:|---:|---:|---:|---|
| Pre-W3.3 baseline (uniform kernel) | 117 | n/a | 0% | 1.000× (reference) | — |
| G23 (1 slice per block) | 468 | 468 | 49.85% | **0.555×** (G24) | Launch overhead dominates |
| G26 Phase 2 (contiguous chunks per block) | 117 | 468 | 0.030% | **0.407×** | Work-balancing collapsed |

**The static-grid design space is empirically exhausted at superhub-50K.** Neither 468-blocks (launch-overhead-bound) nor 117-blocks (work-balance-bound) clears D7a. The intermediate range (234, 256) was implicitly tested by Codex during G26 policy exploration (round_robin/contiguous/reserveK) and none recovered both benefits.

### Why persistent threads is the right next architecture

Persistent threads is the canonical CUDA pattern for irregular workloads where static grid sizing fails. Established in the literature:
* Aila & Laine (2009) "Understanding the Efficiency of Ray Traversal on GPUs"
* Gupta et al. (2012) "A Study of Persistent Threads Style GPU Programming for GPGPU Workloads"
* NVIDIA CUDA Samples: `samples/4_CUDA_Libraries/conjugateGradientCudaGraphs` and various work-pulling examples

**Pattern:** fixed small grid (e.g., `device_sm_count × 2` or `64`). Each block runs a `while(work_remaining)` loop, atomically pulls the next work-unit ID from a device-side counter, processes it, repeats until counter exceeds work-unit count. Result:
- **Single kernel launch** = baseline launch overhead (vs G23's 4×)
- **Dynamic load balancing** = heavy slices don't block light ones (vs G26 Phase 2 collapsed)
- **Work-stealing semantics** = blocks that finish their slice quickly help carry load

### Combined with heavy-slice splitting

Even with work-stealing, a single 5041-row slice is a wall-clock bottleneck (only one block can process it; while it processes, the other 63 blocks run out of work). **The G27 design ALSO splits heavy slices**: any slice with > `MAX_SLICE_OUTPUT_ROWS` (e.g., 256) is decomposed into sub-slices of ≤256 rows. This makes the work-unit-count grow (from 468 → potentially 1000+) but each is bounded.

The combination achieves:
- **Bounded max wall-time per work-unit** (no single slice dominates)
- **Dynamic balancing** (work-stealing distributes the now-uniform-sized units)
- **Baseline launch overhead** (single launch with fixed small grid)

This is the "have your cake and eat it" path — neither static-grid approach can deliver all three properties simultaneously.

---

## G27 — Persistent-threads work-stealing + heavy-slice splitting

### Goal

Cut `feat/w33-persistent-threads-work-stealing` from `feat/w33-slice-aware-implementation @ 6595b969` (NOT G26 Phase 2 which has D7b regression). Implement:
1. **Heavy-slice splitting** in the device-side prefix kernel: post-process slice plan to ensure no slice has > MAX_SLICE_OUTPUT_ROWS output rows; split heavy slices into sub-slices.
2. **Persistent-threads kernels** `wcoj_triangle_count_persistent` + `wcoj_triangle_materialize_persistent` using device-side atomic dispatch counter.
3. **Launch surface** with fixed small grid (`min(device_sm_count × 2, slice_count)` or analogous) + slice_dispatch_counter buffer + reset-before-launch.
4. **Preserve G23 adaptive skew detection**: uniform inputs MUST go through baseline kernel path (D7b stays ≤ ±5%).

Validation: row-equality + D7a ≥ 2.0× at superhub-50K + D7b PASS at uniform-u32-10K + per-block work distribution (informational) + all existing tests + CUDA cert 1/1.

Branch stays unmerged. G28 will be the closure proposal if D7a/D7b both PASS.

### Strategies (GQM+Strategies)

* **S27.1** Cut `feat/w33-persistent-threads-work-stealing` from `feat/w33-slice-aware-implementation @ 6595b969`. Worktree at `.worktrees/w33-persistent-threads`. The G26 branch (`feat/w33-grid-amortized-v2 @ 2aeb74b4`) is preserved as historical evidence of the grid-stride amortization failure mode.

* **S27.2 — Heavy-slice splitting (Step 1).** In `crates/xlog-cuda/src/memory.rs` (the device-side slice-prefix kernel area), add post-processing logic:
  * After the existing weighted-prefix partition produces `slice_starts/slice_ends`, scan each slice's expected output rows.
  * For any slice with `expected_output > MAX_SLICE_OUTPUT_ROWS` (recommend `MAX = 256`; tune empirically): split into ⌈expected/MAX⌉ sub-slices by linearly partitioning the input row range.
  * Update `slice_count` accordingly. Slices that are already ≤MAX stay as-is.
  * Bound: max slice_count after splitting should be 2-4× the original 468 (i.e., 1000-2000 slices reasonable).
  * The split MUST preserve `i` as the original `e_xy` row index inside the kernel body (no output-offset disruption).

* **S27.3 — Persistent-threads kernels (Step 2).** In `crates/xlog-cuda/kernels/wcoj.cu`, add NEW kernel signatures `wcoj_triangle_count_persistent` and `wcoj_triangle_materialize_persistent`. DO NOT modify the existing `_sliced` variants (preserve as historical reference). DO NOT modify the baseline uniform kernels.

  Kernel skeleton (count; materialize analogous):
  ```cuda
  __global__ void wcoj_triangle_count_persistent(
      /* existing kernel params */,
      const uint32_t* __restrict__ slice_starts,
      const uint32_t* __restrict__ slice_ends,
      uint32_t slice_count,
      uint32_t* __restrict__ slice_dispatch_counter
  ) {
      while (true) {
          // Atomically claim next slice in warp leader, broadcast to warp
          uint32_t my_slice;
          if (threadIdx.x == 0) {
              my_slice = atomicAdd(slice_dispatch_counter, 1);
          }
          // Broadcast my_slice from thread 0 to entire block via shared mem
          __shared__ uint32_t shared_slice;
          if (threadIdx.x == 0) shared_slice = my_slice;
          __syncthreads();
          my_slice = shared_slice;

          if (my_slice >= slice_count) break;

          int start = slice_starts[my_slice];
          int end = slice_ends[my_slice];
          for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
              // body IDENTICAL to existing _sliced kernel body (preserves row-equality)
          }
          __syncthreads();  // ensure block completes slice before next claim
      }
  }
  ```

  Note: use `__shared__ uint32_t shared_slice` + `__syncthreads()` broadcast pattern (NOT `__shfl_sync` because the entire block needs the same value, not just within a warp).

* **S27.4 — Launch surface (Step 3).** In `crates/xlog-cuda/src/provider/wcoj.rs`:
  * Choose fixed grid: `let persistent_grid: u32 = std::cmp::min(device_sm_count.saturating_mul(2), slice_count_after_split as u32);` — defaults to ~64-128 for typical Blackwell SM counts. Hard min of 16 to avoid degenerate cases.
  * Allocate slice_dispatch_counter buffer (u32, size 1).
  * Reset to 0 via `cuMemsetD32` or analogous before each launch.
  * Pass counter pointer + slice_starts + slice_ends + slice_count + persistent_grid to kernel.
  * Apply for BOTH count and materialize launches.
  * **CRITICAL:** preserve the existing adaptive skew detection. Uniform inputs (where `skewed == false` per the slice plan flags) MUST still route to BASELINE `wcoj_triangle_count` + `wcoj_triangle_materialize` (NOT persistent). This is the G23 contract that keeps D7b within budget.

* **S27.5 — Adaptive skew detection preservation.** Audit `provider/wcoj.rs` for the `skewed` flag check from G23. The dispatch logic must be: `if skewed { call persistent kernels } else { call baseline kernels }`. Add an integration test if the existing dispatcher doesn't have one: verify uniform-u32-10K invokes `wcoj_triangle_count` (NOT `_persistent`), via kernel-symbol assertion in the design-behavior probe.

* **S27.6 — Acceptance probes update.** Extend `crates/xlog-integration/src/bin/wcoj_design_behavior_probe.rs` to assert:
  * Skewed path invokes `_persistent` kernel symbols.
  * Uniform path invokes baseline (non-`_persistent`) kernel symbols.
  * Row-equality PASS on both.
  * Per-block-output stddev under persistent kernels (informational — work-stealing makes per-block-output less meaningful, but report anyway for comparison).
  * Slice count after heavy-slice splitting reported.
  * Max slice output rows post-splitting (should be ≤ MAX_SLICE_OUTPUT_ROWS).

* **S27.7 — Criterion measurement.** V3 sample_size(200) Criterion at:
  * `superhub-50K` (the D7a binding cell). Required: row-equality PASS + ratio ≥ 2.0×.
  * `uniform-u32-10K` (the D7b spot-check). Required: row-equality PASS + paired delta ≤ ±5%.
  * IF 50K passes, ALSO measure `superhub-200K` for scale-validation evidence.

* **S27.8 — Existing test gates green:**
  * `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run` EXIT 0.
  * `cargo build --release --features wcoj-phase-timing` EXIT 0.
  * `cargo build --release` (no feature) EXIT 0.
  * `cargo fmt --check --all` EXIT 0.

* **S27.9** Branch UNMERGED to all 17 parents (main + G11–G26 + the G26-superseded G25 partial + `6595b969`). No FF-merge, no push, no tag.

* **S27.10** Commit structure (multi-commit allowed; final = README):
  1. `feat(w33): heavy-slice splitting in device prefix kernel`
  2. `feat(w33): persistent-threads kernels (count + materialize)`
  3. `feat(w33): atomic dispatch counter launch surface`
  4. `feat(w33): design probe asserts skewed→persistent + uniform→baseline routing`
  5. Final: `feat(w33): G27 evidence README + closure-readiness verdict`

* **S27.11 — Zero R6 anti-patterns:** `git grep classify_heavy_rows` + `git grep mask_histogram` in W3.3 production code BOTH empty.

### Questions

* **Q27.1** Branch HEAD SHA?
* **Q27.2** Heavy-slice splitting: MAX_SLICE_OUTPUT_ROWS chosen value + file:line? Post-split slice_count + max output rows per slice?
* **Q27.3** Persistent-threads kernels: file:line for `wcoj_triangle_count_persistent` + `wcoj_triangle_materialize_persistent`?
* **Q27.4** Launch surface: persistent_grid value + slice_dispatch_counter alloc + reset; file:line?
* **Q27.5** Adaptive skew detection preserved: uniform path STILL invokes baseline kernel (verified via probe)?
* **Q27.6** D7a superhub-50K under V3: baseline + merge medians + 95% CI + paired delta + ratio + verdict?
* **Q27.7** D7b uniform-u32-10K under V3: paired delta + verdict (should be ≤ ±5% via adaptive skew)?
* **Q27.8** (IF D7a passes) superhub-200K: paired delta + ratio + verdict?
* **Q27.9** Row-equality PASS at all measured scales?
* **Q27.10** All existing tests + CUDA cert 1/1?
* **Q27.11** Branch unmerged from all 17 parents?
* **Q27.12** Zero R6 anti-patterns?

### Metrics

* **M27.1** `feat/w33-persistent-threads-work-stealing` exists; HEAD reachable from none of 17 parents.
* **M27.2** `docs/evidence/2026-05-13-w33-persistent-threads-work-stealing/README.md` with measurement tables.
* **M27.3** `cargo bench --no-run` EXIT 0.
* **M27.4** Slice-bin splitting: max slice output rows post-split ≤ MAX_SLICE_OUTPUT_ROWS (e.g., 256).
* **M27.5** Persistent-threads kernel symbols exist (`wcoj_triangle_count_persistent` + `wcoj_triangle_materialize_persistent`).
* **M27.6** Adaptive skew detection contract: uniform path invokes baseline kernel (verified via design probe + integration test).
* **M27.7** Row-equality PASS at superhub-50K + uniform-u32-10K + (if measured) superhub-200K.
* **M27.8** D7a at superhub-50K under V3: ratio ≥ 2.0×.
* **M27.9** D7b at uniform-u32-10K under V3: paired delta ≤ ±5%.
* **M27.10** All existing tests EXIT 0; CUDA cert 1/1.
* **M27.11** Both feature on/off builds compile; fmt EXIT 0.
* **M27.12** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "feat/w33*"` empty.
* **M27.13** Branch unmerged from all 17 parents.
* **M27.14** Zero R6 anti-patterns.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse feat/w33-persistent-threads-work-stealing` ≠ all 17 parent SHAs.
* Verify M27.6 — uniform path STILL invokes baseline kernel symbols.
* Run both feature builds + fmt + CUDA cert from supervisor session.
* Verify D7a ≥ 2.0× at superhub-50K AND D7b ≤ ±5% at uniform-u32-10K.
* Verify branch unmerged + no tag + no origin push.

If D7a + D7b both PASS: G28 = closure proposal grounded in G11–G27 evidence chain + board OPEN → DONE + memory + MEMORY.md + FF-merge of `feat/w33-paper-aligned-plan-it1` to main (per W2.5/W4.2/W4.3/W5.2 precedent).

If D7a passes but D7b fails: G28 = focused fix on adaptive skew detection (likely a launch-surface bug).

If D7a fails: G28 = deeper RCA on the persistent-threads architecture (likely heavy-slice split threshold tuning OR fundamental architecture revisit). Per user directive, NO defer.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge into main.
* No `docs/v065-closure-board.md` edit (G28's conditional job).
* No `v0.6.6` references.
* **No R6 anti-patterns** (per-call histogram launch / heavy-light kernel split / per-call classify_heavy_rows / front-end mask_histogram+classify+partition_scan; each measured-rejected per `f1142b3e`).
* No modification of baseline kernels `wcoj_triangle_count` + `wcoj_triangle_materialize` (uniform path must stay bit-identical).
* No modification of existing `_sliced` kernels (preserve as historical reference for G23/G26).
* No removal of row-equality assertions.
* No D7 amendment.
* No closure proposal in this goal.
* **No simplification** of the persistent-threads design (single-launch + atomic counter + heavy-slice splitting are all required).
* **No silent bypass** of the skewed/uniform routing — the adaptive skew detection contract is non-negotiable.

### Why this is the closure-attempt goal

26 prior goals have explored the W3.3 design space: paper-aligned plan, scaffolding-stub identification, slice-aware kernels, measurement-noise attribution (×4), launch-overhead-vs-work-balancing tradeoff, grid-stride amortization (failed), and now persistent-threads work-stealing. G27 has all prior empirical findings as inputs: it knows what doesn't work (static-grid both ways) and applies a fundamentally different architecture (dynamic work-stealing + bounded slice sizes) that addresses both prior failure modes simultaneously.

Per the user's "no defers, no toyshit" directive: this is the proper structural fix. If it fails D7a, the next step is deeper RCA or alternative architecture — NOT deferral.

Proceed: cut `feat/w33-persistent-threads-work-stealing` from `6595b969`, execute the 4-step implementation (slice splitting + persistent kernels + atomic-counter launch + probe assertions), run V3 Criterion at 50K + 10K + (conditional) 200K, write evidence README with closure-readiness verdict, multi-commit allowed (final = README). Emit REVIEW REQUEST with HEAD SHA + per-cell measurements + D7a verdict + D7b verdict + closure-readiness recommendation.
