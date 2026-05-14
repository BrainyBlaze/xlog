# Supervisor Goal 023 — W3.3 Slice-Aware Implementation (Execute G22 Fix Recipe)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G22 design-behavior RCA-4 APPROVED. Forensic commit `258bddc6` on `forensic/w33-design-behavior-rca4`. **Confirmed root cause: hybrid RC1+RC3.** W3.3 merge-resident design is currently a scaffolding stub — `black_box(launch_slices)` at `crates/xlog-cuda/src/provider/wcoj.rs:231-233` discards computed metadata before kernel launch; the "histogram" at `crates/xlog-cuda/src/memory.rs:1165-1186` is just `ceil(rows/256)` with no real bin vector; kernels at `crates/xlog-cuda/kernels/wcoj.cu:240-270` + `:309-348` use textbook uniform `blockIdx.x*blockDim.x+threadIdx.x` mapping with NO slice parameters. Per-block-output stddev variance reduction is **0.000000%** despite the fixture being genuinely 24× skewed (max=5,203 vs median=211).
**Date:** 2026-05-12.

---

## Context

User directive recorded 2026-05-12 (verbatim): *"NO ANY FUCKING DEFERS, TOYSHIT, SIMPLIFICATION. SETUP A CLEAR GOAL TO FIX THE ROOT CAUSE AND DELIVER FULL AND ROBUST IMPLEMENTATION!!!!"*

G23 is the largest production-code change in the 22-iteration W3.3 chain. It executes G22's 6-step fix recipe verbatim. This is real engineering work: new device-resident slice plan, slice-aware kernel signatures, slice-driven launch grid, Merge-phase refresh wiring. **No shortcuts, no simplifications, no toy implementations.**

### Discipline anchors for this goal

* **No R6 anti-patterns** (per W3.3 plan iteration 1 Forbidden Directions @ `feat/w33-paper-aligned-plan-it1 @ a4c299fd`): no per-call histogram launch, no heavy/light kernel split, no per-call `classify_heavy_rows` kernel, no front-end `mask_histogram`/`classify`/`partition_scan` pass. Each of those is `measured-rejected per f1142b3e`. The G23 design uses **paper-P3 launch-time slice consumption**, not front-end histogram.
* **Paper P3 alignment retained** (verbatim from `reference_srdatalog_paper.md`): *"Histograms maintained alongside data; computed incrementally during Merge; consumed at kernel launch-time to assign balanced thread-block work-unit slices over the outermost-join's search space."* — G23 implements ALL THREE clauses for the first time.
* **D7a + D7b gates LOCKED** per W3.3 plan iteration 1 D5: D7a remains `≥ 2.0×` speedup on superhub-50K; D7b remains `±5%` uniform-u32-10K regression budget. No amendment.
* **Row-equality is non-negotiable.** Every existing W3.3 test + the CUDA cert suite + the new G23 acceptance probes must produce row-equal output to baseline.

---

## G23 — Slice-aware implementation

### Goal

Cut `feat/w33-slice-aware-implementation` from `forensic/w33-design-behavior-rca4 @ 258bddc6` (preserves G22 forensic + the feature-gated diagnostic probes). Implement the 6-step fix recipe verbatim. The implementation must:

1. Store a real device-resident slice plan (slice_starts + slice_ends arrays).
2. Compute slice bins from the outermost-join search space during Merge-phase refresh.
3. Pass the slice plan to the kernel launch (no more `black_box`).
4. Launch with `grid_dim.x = slice_count` (not `ceil(rows/256)`).
5. Extend kernel signatures with slice parameters AND kernel bodies that consume them.
6. Preserve output offsets indexed by original `e_xy` row.

Branch stays unmerged until G24 validates speedup; G25 will then FF-merge `feat/w33-paper-aligned-plan-it1` (the plan) — NOT this implementation branch directly — into main per the W2.5/W4.2/W4.3/W5.2 closure precedent (closure proposal commit lands on the plan branch, then FF-merge).

### Strategies (GQM+Strategies)

* **S23.1** Cut `feat/w33-slice-aware-implementation` from `forensic/w33-design-behavior-rca4 @ 258bddc6`. Worktree at `.worktrees/w33-slice-aware`.

* **S23.2 — Slice plan storage** (Step 1 of recipe).
  * In `crates/xlog-cuda/src/memory.rs:1052-1062`, replace the current metadata-only struct with a real slice plan structure on `CudaBuffer` or sibling. Required fields:
    * `slice_starts: TrackedCudaSlice<u32>` (device-resident, per-block start row index)
    * `slice_ends: TrackedCudaSlice<u32>` (device-resident, per-block end row index, exclusive)
    * `slice_count: usize` (host-side)
    * `logical_rows: usize` (preserved for compatibility)
    * `refresh_phase: WcojHistogramRefreshPhase` (preserved)
  * The slice plan MUST be relation-tied (lifetime equals the `CudaBuffer` lifetime) — paper-P5 flat-columnar contract.
  * Existing `wcoj_merge_resident_histogram_for_spike` accessor MUST be renamed or extended to expose the slice plan; the feature-gated diagnostic probes that G22 added MUST continue to compile and produce meaningful output.

* **S23.3 — Refresh body** (Step 2 of recipe).
  * In `crates/xlog-cuda/src/memory.rs:1165-1186`, replace the trivial `root_slices = ceil(logical_rows / 256)` with a real bin-computation pass that:
    * Inspects the outermost-join sort key column of the merged relation.
    * Computes balanced slice boundaries such that each block processes approximately equal *output rows* (NOT input rows — the fixture's skew is in output cardinality per input row).
    * The bin count should target approximately `sqrt(logical_rows)` or follow a profile-driven heuristic that produces measurably more balanced slices than uniform `ceil(rows/256)`. The exact heuristic is the implementer's call provided it (a) produces device-resident slice_starts/slice_ends arrays, (b) demonstrably reduces per-block-output stddev vs baseline (acceptance probe in step 6), and (c) executes deterministically with the same inputs.
  * Implementation pattern: a small CUDA kernel that scans the input row range and emits prefix sums of *expected output rows per input row* (e.g., based on `e_yz` neighbor count for each `e_xy` row's y-value), then a second pass that partitions the prefix-sum into balanced slices. The reduction pass MUST stay within the existing flat-columnar + deterministic-offsets contract (paper P5).
  * In `crates/xlog-runtime/src/executor/recursive.rs:398-399` and `:725-727`, wire the refresh call to use the new bin computation. The call site stays the same; the function body becomes substantive.

* **S23.4 — Launch surface** (Step 3 + 4 of recipe).
  * In `crates/xlog-cuda/src/provider/wcoj.rs:231-233`, REMOVE the `black_box(launch_slices)` discard. Replace with real slice-plan retrieval + passing.
  * In `crates/xlog-cuda/src/provider/wcoj.rs:837-878` (count launch) and `:1025-1075` (materialize launch):
    * Replace `grid = ceil(n_xy / 256)` with `grid = slice_count`.
    * Add `slice_starts`, `slice_ends`, `slice_count` to the kernel call parameter list.
    * Preserve `LaunchRecorder` semantics; the new parameters are device pointers, recorded normally.

* **S23.5 — Slice-aware kernel** (Step 4 of recipe — continued).
  * In `crates/xlog-cuda/kernels/wcoj.cu:240-270` (count) and `:309-348` (materialize):
    * Add new kernel signatures `wcoj_triangle_count_sliced` and `wcoj_triangle_materialize_sliced` — DO NOT modify the existing `wcoj_triangle_count` and `wcoj_triangle_materialize` (which are still used by baseline path and existing tests).
    * The sliced kernel body uses: `int start = slice_starts[blockIdx.x]; int end = slice_ends[blockIdx.x]; for (int i = start + threadIdx.x; i < end; i += blockDim.x) { ... }`.
    * The body of the for-loop is IDENTICAL to the uniform kernel's body — only the iteration bounds change. This guarantees row-equality.
    * Both `count_sliced` and `materialize_sliced` receive the SAME slice plan (consistency).
* **S23.6 — Output offset preservation** (Step 5 of recipe). The `i` variable inside the sliced kernel body MUST remain the original `e_xy` row index. The count kernel writes `out_counts[i]`; the materialize kernel reads `offsets[i]`. This preserves the existing count→scan→materialize pipeline.

* **S23.7 — Acceptance probes** (Step 6 of recipe).
  * Extend the G22 design-behavior probe binary `crates/xlog-integration/src/bin/wcoj_design_behavior_probe.rs` to ALSO check after this fix:
    * Kernel-symbol/parameter delta NON-EMPTY (proves slice plan reaches kernel).
    * Per-block-output stddev on superhub-50K MUST be strictly less than baseline's 459.998 (proves slicing balances work).
    * Row-equality PASS (correctness).
  * Add a feature-gated assertion in the merge-resident launch path that runs ONCE per process under `wcoj-phase-timing`: if the launched grid_dim equals the baseline `ceil(rows/256)`, panic with a clear message. This prevents future scaffolding-stub regressions.

* **S23.8 — Existing test preservation.** Every existing test that was green pre-G23 must remain green:
  * `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1 (the canonical CUDA cert).
  * Targeted W3.x tests: existing slice-4 stable-triangle counter remains exact `== 1` on bare default; W3.1/W3.2 layout-sort tests unchanged.
  * D7b spot-check: uniform-u32-10K under V3 protocol (sample_size(200)) paired delta must stay within ±5%.

* **S23.9 — Branch hygiene.** Branch UNMERGED to all 13 parents (main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed, G19-scale-sweep, G20-stability-rca3, G21-scale-sweep-v3-stable, G22-design-behavior-rca4). No FF-merge, no push, no tag.

* **S23.10 — Commit structure.** Multiple commits ARE allowed for G23 because of the size, but the final commit must be either an evidence README OR a comprehensive acceptance-probe run output. Suggested structure:
  1. `feat(w33): slice plan storage on CudaBuffer (memory.rs)`
  2. `feat(w33): outermost-join slice-bin computation in Merge refresh (memory.rs + recursive.rs)`
  3. `feat(w33): slice-aware launch surface (provider/wcoj.rs)`
  4. `feat(w33): wcoj_triangle_{count,materialize}_sliced kernels (wcoj.cu)`
  5. `feat(w33): G23 acceptance probes + regression-prevention assertion`
  6. `feat(w33): G23 evidence run (existing tests pass + per-block stddev improved + row-equality PASS)`

### Questions

* **Q23.1** Branch HEAD SHA?
* **Q23.2** Slice-plan storage: what struct/fields/lifetime did you choose? Cite file:line.
* **Q23.3** Slice-bin computation: what heuristic? Show the CUDA kernel + prefix-sum partition logic with file:line citations.
* **Q23.4** Launch surface: grid_dim.x = slice_count confirmed at provider/wcoj.rs file:line?
* **Q23.5** Slice-aware kernel: new `_sliced` symbols added; for-loop bounds correctly use slice_starts/slice_ends?
* **Q23.6** G23 acceptance probe: per-block-output stddev on superhub-50K. Baseline = 459.998, merge-resident-sliced = ? quantify reduction.
* **Q23.7** Row-equality: PASS on superhub-50K + all existing W3.x tests + CUDA cert suite (1/1)?
* **Q23.8** D7b spot-check uniform-u32-10K under V3 protocol: paired delta + verdict?
* **Q23.9** R6 anti-pattern check: zero per-call histogram launches, zero heavy/light kernel split, zero front-end `classify_heavy_rows`?
* **Q23.10** Branch unmerged from all 13 parents?

### Metrics

* **M23.1** `feat/w33-slice-aware-implementation` exists; HEAD reachable from none of 13 parent branches.
* **M23.2** Slice plan struct contains `slice_starts: TrackedCudaSlice<u32>`, `slice_ends: TrackedCudaSlice<u32>`, `slice_count: usize`.
* **M23.3** Slice-bin computation in `memory.rs` actually inspects column contents (not `ceil(rows/256)`). Verifiable via diff.
* **M23.4** `provider/wcoj.rs` no longer contains `black_box(launch_slices)`. Verifiable via `git grep`.
* **M23.5** New `_sliced` kernel symbols exist in `wcoj.cu`. Verifiable via `git grep wcoj_triangle_count_sliced` and `git grep wcoj_triangle_materialize_sliced`.
* **M23.6** Both `_sliced` kernel bodies use slice_starts/slice_ends in for-loop bounds.
* **M23.7** Acceptance probe: per-block-output stddev on superhub-50K under merge-resident-sliced **strictly less than 459.998** baseline.
* **M23.8** Row-equality PASS: superhub-50K, all existing W3.x tests, CUDA cert suite 1/1.
* **M23.9** D7b spot-check uniform-u32-10K paired delta ≤ ±5%.
* **M23.10** `cargo build --release --features wcoj-phase-timing` EXIT 0.
* **M23.11** `cargo build --release` (no wcoj-phase-timing) EXIT 0.
* **M23.12** `cargo fmt --check --all` EXIT 0.
* **M23.13** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "feat/w33*"` empty.
* **M23.14** Branch unmerged from all 13 parents.
* **M23.15** Zero R6 anti-patterns introduced (`git grep classify_heavy_rows`, `git grep mask_histogram` in new W3.3 code yields no results).

### Supervisor validation per locked protocol

* Read every G23 commit message.
* `git rev-parse feat/w33-slice-aware-implementation` ≠ all 13 parent SHAs.
* Verify M23.4 (no `black_box(launch_slices)`), M23.5 (new `_sliced` symbols), M23.6 (slice-aware bodies) by inspection.
* Verify M23.15 (no R6 anti-patterns) by `git grep`.
* Run `cargo build --release` AND `cargo build --release --features wcoj-phase-timing` from supervisor session; both EXIT 0.
* Run the G23 acceptance probe; verify per-block-output stddev < 459.998 and row-equality PASS.
* Run the CUDA cert suite from supervisor session: 1/1.
* Run D7b spot-check; verify paired delta within ±5%.
* Verify branch unmerged + no tag + no origin push.

If all M23 metrics green: G24 = scale-sweep benchmark on the implementation under V3 protocol (50K + 200K + 1M) to measure actual D7a speedup. G25 = closure proposal + board OPEN → DONE + memory file + MEMORY.md update + FF-merge of `feat/w33-paper-aligned-plan-it1` to main (per W2.5/W4.2/W4.3/W5.2 precedent).

If any M23 metric fails: report specifically which + adjust scope; user-decision NOT yet required because G23 is engineering, not measurement.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `feat/w33-slice-aware-implementation` into main in this goal.
* No `docs/v065-closure-board.md` edit (G25's job, conditional on G24).
* No `v0.6.6` references.
* **No R6 anti-patterns:** no per-call histogram launch, no heavy/light kernel split, no per-call `classify_heavy_rows` kernel, no front-end `mask_histogram`/`classify`/`partition_scan` pass. Each is `measured-rejected per f1142b3e`.
* No modification of existing `wcoj_triangle_count` or `wcoj_triangle_materialize` kernels (the baseline uniform path). Add NEW `_sliced` variants instead. Baseline must remain bit-identical for non-merge-resident invocations.
* No removal of row-equality assertions anywhere.
* No D7 amendment.
* No closure proposal in this goal.
* No "simplification" of the slice-bin computation to anything that produces no measurable variance reduction. The acceptance probe M23.7 enforces this.
* No new toy/stub implementation; the slice plan MUST be device-resident and MUST drive actual kernel-launch grid sizing.

### Why this is the chain's load-bearing moment

22 supervisor goals + ~13 codex iterations have produced the precondition for this goal: a file:line-cited root cause + a 6-step fix recipe + measurable acceptance criteria. Every prior goal was either planning, spike, forensic, or measurement-attribution. G23 is the **first goal that ships actual production code intended to deliver the W3.3 speedup**. The chain's value cashes out here.

If G23 lands clean (M23.7 stddev < 459.998, row-equality PASS, no R6 anti-patterns, both feature builds compile), the path to W3.3 closure is mechanical: G24 measures speedup at scale, G25 writes closure. If G23 fails any acceptance gate, we have a precise diagnostic of which step of the 6-step recipe didn't deliver — and the chain continues with concrete data, not abstract uncertainty.

Proceed: cut `feat/w33-slice-aware-implementation` from `258bddc6`, execute the 6-step fix recipe verbatim with the constraints above, multi-commit allowed but final commit must include G23 acceptance probe run output, emit REVIEW REQUEST with HEAD SHA + per-block stddev numbers + row-equality verdicts + D7b spot-check + zero-R6-anti-pattern confirmation.
