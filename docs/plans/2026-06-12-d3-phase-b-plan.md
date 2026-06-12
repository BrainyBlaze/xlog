# D3 Phase B — Production Integration of the Factorized Recursive Delta

Date: 2026-06-12. Precondition: S3 gate PASS (`docs/evidence/2026-06-12-s3-factorized-delta/`,
40.03×/59.95× peak-memory at 0.087×/0.071× wall-clock). Branch `feat/d3-factorized-delta`.
Authorized by @human in #xlog (msg-20260612-192125).

## 1. Goal and Definition of Done

Route qualifying semi-naive delta steps in the production recursive engine through
`fj_delta_novel_u32_recorded`, fail-closed to the existing hash-join→diff path everywhere
else. DoD:

1. TC-shaped recursive rules dispatch the factorized path in `execute_recursive_scc`
   (`factorized_delta_dispatch_count ≥ 1`) with row-set parity vs kill-switch fallback.
2. Kill switch `XLOG_DISABLE_FACTORIZED_DELTA=1` forces the legacy path (process-global,
   like `XLOG_DISABLE_FREE_JOIN`).
3. Counter surfaced on `Executor`, in `EpistemicGpuRuntimeCounters` (+ both delta fns,
   snapshot, CLI evidence JSON) — mirror of `free_join_dispatch_count`.
4. Declines are silent and complete: non-ChainJoin bodies, >1 delta occurrence in the
   node, non-u32/Symbol schemas, arity ≠ 2, out-of-cap domain, per-iteration tiny-delta
   heuristic — all fall back with counter untouched and identical results.
5. Full local regression green; production-dispatch bench guard (Step-12 style) on RunPod:
   engine ON vs OFF on the dense gate fixture (expect win) AND a sparse long-chain fixture
   (expect ≤1.2× — the heuristic must bail). Merge decision only after that evidence.

## 2. Recognition (empirically grounded)

Probe (2026-06-12, this branch): `q(X,Z) :- q(X,Y), edge(Y,Z)` compiles to
`ChainJoin{ left: Scan(q), right: Scan(edge), left_key: 1, right_key: 0,
output_columns: [Column(0), Column(3)], fallback: Project{Join{...}} }`, and
`rewrite_scan_nth` swaps the delta RelId inside ChainJoin variants coherently
(`rewrite.rs` same-occurrence contract). Recognition therefore matches **ChainJoin with
two Scan children where exactly one scans the delta relation**:

- delta side: key col = that side's chain key; carry col = the other column (arity 2).
- static side: key col, value col = the other column.
- `output_columns` must be `[Column(i), Column(j)]` with `{i,j}` = {delta-carry,
  static-value} **in the combined left+right column space**, either order (order flag
  decides final column placement).
- Static side may be the *full* recursive predicate itself (non-linear self-join
  `q(X,Z) :- q(X,Y), q(Y,Z)` variants) — supported; its buffer is re-layout-normalized
  per iteration (R is maintained sorted+deduped by `union_gpu`, so the sorted+unique
  fast path applies).

## 3. Execution plan per dispatch

1. Static side: layout-normalize. Key col 0 → `wcoj_layout_u32_recorded` directly;
   key col 1 → column-swap projection first.
2. Provider entry generalized to take explicit column indices for delta (carry, key) and
   full-R (carry, value), plus an output-order flag — no data copies for orientation,
   kernels unchanged.
3. Novel returned in head-column order; merged per head:
   - all contributions factorized → `delta_new` = novel (union of novels if several
     rules) — **diff is skipped entirely** (novel ∩ R = ∅ by construction);
   - mixed factorized + raw → novel folded into the raw accumulator, existing
     `diff_gpu` finalize runs unchanged (diff is a no-op on novel rows — sound).
4. Union into full and store/stats updates unchanged.

## 4. Gating (dense-domain, conservative, env-tunable)

- Domain = max(static cols, base-R₀ cols) + 1, computed **once per SCC fixpoint** with a
  new trivial `fj_delta_max_u32` atomicMax kernel (induction: every derived id comes from
  base ∪ static columns). In-kernel bounds checks remain as fail-closed backstop.
- Caps: default dispatch cap `domain ≤ 2^14` (bitmap 32 MiB + counts 128 MiB),
  `XLOG_FACTORIZED_DELTA_MAX_DOMAIN` override, hard bound 2^16 (provider invariant).
- Per-iteration heuristic (protects sparse/long-chain workloads from the
  popcount+scan floor over `n_words = domain²/32`): dispatch iff
  `|delta| × max(1, |static| / domain) ≥ n_words / 8` (divisor env-tunable,
  `XLOG_FACTORIZED_DELTA_WORK_DIVISOR`). Iterations that bail use the legacy path —
  mixing paths across iterations is sound (both produce the exact novel set).
- Sparse-domain representation remains explicitly deferred (design §8) — this phase
  ships dense-domain only, with the gate above keeping it off elsewhere.

## 5. Steps

1. `feat(cuda)`: `fj_delta_max_u32` kernel + manifest/consts; provider entry
   generalization (column indices + output-order flag); column-max helper.
2. `feat(runtime)`: `try_dispatch_factorized_delta` + integration into
   `execute_recursive_scc` variant loop and per-head finalize; kill switch; counter;
   epistemic counter plumbing + CLI evidence field.
3. `test(integration)`: e2e suite — right-linear TC fires + kill-switch parity;
   non-linear self-join TC; left-linear; swapped-head; declines (u64, 3-atom, domain
   over cap) with parity; existing spike parity suite stays green.
4. Local regression: workspace sweep.
5. RunPod bench guard (one minimal pod, D2 recipe with `--startSSH`): engine ON vs OFF —
   dense gate fixture and sparse long-chain fixture; record both; ≤1.2× on sparse is the
   no-regression bar; pod deleted + confirmed.
6. Evidence + CHANGELOG `[Unreleased]` + merge decision posted to #xlog.
