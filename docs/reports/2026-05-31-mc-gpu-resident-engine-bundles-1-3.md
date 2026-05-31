# Supervisor Report — GPU-Resident Datalog/MC Engine (Bundles 1–3)

**Date:** 2026-05-31
**Branch:** `feat/mc-gpu-resident-engine` (worktree `.worktrees/mc-gpu-resident-engine`)
**HEAD:** `dcfc6f1c` — based on `main` / v0.9.0-rc. Working tree clean, unpushed.
**Overall verdict:** **MERGE_CANDIDATE** — the goal's hard requirements and the
explicit minimum-acceptance list are met and independently re-verified. Two
non-blocking cleanups noted at the end.

---

## Bundle 1 — MC Zero-Host Hot-Loop Cleanup (precursor; `a894aab4`)

**Status: superseded precursor.** Its mechanism no longer exists in the tree (the
host loop it patched was deleted in Bundle 2/3). Recorded for lineage.

- **Scope:** the original host-orchestrated MC loop re-uploaded query/evidence
  row-count *pointer arrays* every sample (one tracked HtoD/sample).
- **Fix:** engine-owned stable row-count buffers + once-before-loop pointer upload
  + per-sample `dtod_copy` (device→device, untracked). Pre-fix measurement: 256
  HtoD / 2048 B for 256 samples; post-fix 0 tracked HtoD/DtoH in the loop.
- **Honest scoping correction made during review:** "zero" there meant zero
  *tracked data-plane* transfers; the loop still did per-sample host orchestration
  and untracked `dtoh_scalar_untracked` row-count reads. That is exactly why it was
  only a precursor and did not close the real goal.

## Bundle 2 — GPU-Resident dense-boolean engine (MVP; `d4de0201`→`b1451eb1`)

**Status: bounded-fragment precursor, retained as a device-side membership/dedup
index inside Bundle 3.** Not the final proof surface.

- **Engine boundary:** one `mc_resident_engine` megakernel; world id = CUDA grid
  dim (one block/world); device-side double-buffered naive fixpoint with a shared
  change flag; on-device counting. Host does one launch + post-sync; reads
  aggregates only after.
- **What it proved:** the no-host loop is achievable end-to-end. Measured-region
  `McNoHostStats` all zero, constant across N=128 and N=1024. Added the
  `untracked_metadata_dtoh_count` provider counter so untracked reads are proven
  zero, not assumed.
- **Rewire + legacy deletion:** `evaluate_gpu_device*` routed solely through the
  resident engine (no fallback); deleted the legacy host-loop Rust
  (`evaluate_gpu_counts_with`, `build_gpu_plan`, `sampling.rs`, dead `buffers.rs`)
  — build warning-free.
- **Limitation that kept it a precursor:** dense `domain^arity` representation →
  bounded domain, arity ≤ 2, body ≤ 2. Supervisor correctly rejected this as the
  closure because it doesn't reuse the sparse relational/WCOJ surface.

## Bundle 3 — WCOJ / tensorized world-batched sparse engine (deliverable; `174a44b4`→`dcfc6f1c`)

Design-checkpoint-first (`docs/plans/2026-05-31-wcoj-world-batched-mc-engine.md`),
then implemented.

### Engine boundary
- World/sample id is a first-class relation dimension. Relations are sparse
  world-segmented columnar arenas — `slot, arg0, arg1, arg2` columns + device row
  counts, static device offsets, convergence flags, overflow flags,
  block-participation counters.
- WCOJ-style positive joins bind from sparse body rows and append derived heads
  through device atomic cursors — no host count-readback in operator chaining.
- Recursive fixpoint orchestrated device-side inside the single measured launch
  (device change flags). Multi-block-per-world is opt-in via
  `XLOG_MC_RESIDENT_BLOCKS_PER_WORLD>1` using cooperative launch + fenced
  cooperative barriers + atomic device change/continue reads.
- Dense bitset demoted to a per-world device-side membership/dedup index.
- Host launches a bounded engine before the measured region and reads final
  aggregates after — nothing in between.

### Supported vs rejected fragment
- **Supported (structural, never predicate-name):** bounded positive relational
  Datalog/MC — arity ≤ 3, body ≤ 3, vars ≤ 8, finite ground universe,
  probabilistic facts, ADs, evidence, queries.
- **Rejected fail-closed before execution (typed):** negation,
  comparison/arithmetic/univ, arity/body over limits, over-budget resident
  arenas. No CPU fallback, no host-sizing fallback.

### No-host instrumentation (measured-region, asserted exactly)
```
McNoHostStats { tracked_htod_calls: 0, tracked_dtoh_calls: 0,
  untracked_metadata_reads: 0, engine_launches: 1, host_loop_iterations: 0,
  per_sample_host_launches: 0, host_fixpoint_iterations: 0,
  per_operator_host_allocations: 0 }
```
Asserted via `is_no_host()` (all eight counters); shown constant in N.

### Recursive fixpoint evidence
- Recursive generic closure: exact counts `[samples, samples, 0]`, `iter_trace == 4`
  all worlds, sparse rows == 9.
- Diagnostic recursive program: `path_rel(1,4)` exact across all 48 worlds;
  converged flags all 1, overflow all 0, participation all 1.
- Cooperative multi-block pilot: counts `[8,0]`, `iter_trace [4;8]`, sparse rows
  `[9;8]`, converged `[1;8]`, overflow `[0;8]`, participation `[2;8]`.
- Non-base derived tuples produced via device-side fixpoint (`iter_trace>1`).

### Device-resident sizing & over-budget fail-closed
- Worst-case arena bounds checked before device allocation when
  `XLOG_MC_RESIDENT_MEMORY_BUDGET_BYTES` is set; over-budget → typed
  `resident_resource_budget` diagnostic carrying `bound_bytes` + `budget_bytes`,
  before execution. Not overclaimed as automatically practical — it fails closed.

### Changed files (Bundle 3)
`crates/xlog-cuda/kernels/mc_resident.cu`, `crates/xlog-cuda/src/cuda_compat.rs`,
`crates/xlog-cuda/src/memory.rs` (alloc counter),
`crates/xlog-prob/src/mc/resident.rs`, `crates/xlog-prob/tests/mc_resident.rs`,
`docs/architecture/xlog-prob.md`,
`docs/plans/2026-05-31-wcoj-world-batched-mc-engine.md`, `CHANGELOG.md`,
`ROADMAP.md`, `examples/prob/04-recursive-mc.xlog` (+ removed
`04-nonmonotone-mc.xlog`), `docs/plans/2026-05-31-mc-gpu-resident-agent-dispatch.md`.

---

## Independently re-verified gates (RTX PRO 3000 Blackwell, against `dcfc6f1c`)

| Gate | Result |
|---|---|
| `mc_resident` | 22 passed (sparse single/3-way/ternary joins, sparse rule-chaining, over-budget fail-closed, dense pilots) |
| `mc_gpu_native` | 9 passed |
| `gpu_mc_device_counts` | 4 passed |
| `epistemic_prob_gpu_accepted_evidence` | 31 passed |
| `epistemic_prob_production_reuse` | 7 passed |
| `cargo fmt --check` | passed |
| `git diff --check` | passed |
| conflict-marker scan | no matches |

**Anti-gaming spot-check (passed):** pilots assert `is_no_host()` (all 8 zeros) and
exact counts (`assert_eq!(counts[…], samples)`, sparse offsets) — not non-empty;
the over-budget test panics if the run returns `Ok` and asserts the typed
`resident_resource_budget` / `bound_bytes` / `budget_bytes` diagnostic.

## Acceptance checklist (supervisor minimum)
- [x] Existing dense no-host pilots remain green.
- [x] New sparse/WCOJ world-batched pilot proves no host interaction (constant in N).
- [x] Recursive transitive closure via device-side fixpoint produces non-base tuples.
- [x] Instrumentation reports all required zeros.
- [x] Negative over-budget WCOJ bound fails closed before measured execution.
- [x] Docs state dense = bounded fragment, WCOJ world-batched = real general path.

## Remaining (non-blocking)
1. Conservative worst-case arena bounds may reject large programs; AGM/tighter
   bounds + broader operator reuse are future scalability work. Fail-closed is the
   accepted behavior; does not affect the gated supported fragment.
2. Dead `mc_eval.cu` primitives (old per-sample truth/accumulate) still present,
   reachable only by 2 isolated kernel-unit tests in `mc_gpu_native.rs` — on no
   execution path; trivial follow-up to delete.

## Commit graph (this work)
```
dcfc6f1c  Bundle 3: sparse WCOJ world-batched engine          <- deliverable
d128bf23  Bundle 3: no-host instrumentation foundation
174a44b4  Bundle 3: design checkpoint + example/doc fixes
b1451eb1  Bundle 2: K7 docs
246ec490  Bundle 2: delete legacy host-orchestrated Rust
d6b91e12  Bundle 2: CHANGELOG
834dfc4d  Bundle 2: route evaluate_gpu_device* -> resident (no fallback)
d4de0201  Bundle 2: dense megakernel + K1-K5 pilots
a894aab4  Bundle 1: zero-tracked-transfer hot loop (precursor)
```

**Recommendation:** Bundle 3 is implemented and independently verified against
every acceptance criterion → MERGE_CANDIDATE, pending supervisor confirmation.
