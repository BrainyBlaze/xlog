# W4.3 Sort-Merge Join — Read-Only Recon + Spike Proposal

**Worktree:** `.worktrees/w43-sort-merge-join` on branch `feat/w43-sort-merge-join` (off local `main` `19f7bc5d`).
**Date:** 2026-05-08.
**Status:** Pre-plan recon. No code yet, no spike yet, no plan-iteration yet. The decision items below are the inputs the iteration-1 plan will lock.

## Board acceptance gate (locked)

From `docs/v065-closure-board.md`:

> W4.3 | ROADMAP item #15 | OPEN | — | General sort-merge join operator for pre-sorted binary relations. Triangle-layout helper is a special case; this is the generic path. | Cert: pre-sorted binary join skips the sort step, matches reference output.

The acceptance gate is functional, not perf-locked: the cert proves "skips sort + matches reference". Perf is not in the acceptance grid by board wording — but a perf-spike is still warranted per `feedback_perf_bench_spike_first.md` because the IMPLEMENTATION DECISION (do we ship a sort-merge operator at all? do we wire it for production dispatch?) hinges on whether the operator is fast enough to be useful.

## Recon findings (4 surface questions)

### 1. Kernel surface

* `crates/xlog-cuda/kernels/sort.cu` — full radix-sort family (histogram → digit-prefix-sums → ranks → scatter, plus permutation appliers + key-gather kernels). NO sort-merge join kernel.
* `crates/xlog-cuda/kernels/wcoj.cu` carries `wcoj_layout_check_sorted_unique_u32/u64` (W3.2's runtime-detection kernels) — useful precedent for a "scan and decide" sortedness check.
* W4.3 needs a NEW kernel: a sort-merge join kernel that takes two pre-sorted single-key buffers and emits matched `(left_idx, right_idx)` pairs (similar emit-pairs design to W4.2's nested-loop kernel).

### 2. Provider surface

* `crates/xlog-cuda/src/provider/relational.rs::sort` (line 1459) — `pub fn sort(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer>`. Sorts a buffer by the given key columns.
* `dedup` (line ~242) calls `sort` + `dedup_sorted` — established pattern of sort-then-do-something.
* No existing `sort_merge_join_*` provider entry point.
* W4.3 needs `pub fn sort_merge_join_v2_inner_u32_1key(left, right, left_key, right_key) -> Result<CudaBuffer>` (signature mirroring W4.2's nested-loop fn for symmetry).

### 3. Dispatch surface (the key design question)

How does production runtime know inputs are pre-sorted? Three tractable options:

| Option | Description | Cost | Safety | Where it fits |
|--------|-------------|------|--------|---------------|
| **A. Caller-asserted flag** | API: `sort_merge_join_v2(left, right, lk, rk, assume_sorted: bool)` | O(1) | Caller must guarantee — UB on misuse | Test fixtures / explicit callers |
| **B. Runtime detection kernel** | Scan inputs, return `is_sorted_by_key`. Same pattern as W3.2's `wcoj_layout_check_sorted_unique_*`. | O(n) per side, but bounded by 1 D2H scalar read | Self-checking — fail-closed on unsorted | Production dispatch via `execute_join` |
| **C. Producer-tracked metadata** | `CudaBuffer.sort_status: Option<Vec<usize>>` set by `sort()`'s output, consumed by joins | O(1) | Provider-level — guaranteed correct if propagated | Most invasive; requires schema/buffer changes |
| **D. IR-level annotation** | `RirNode::Sort` operator + downstream lowerer threading | O(0) at runtime; compile-time only | Sound if compile-time analysis is correct | Most invasive; major IR change |

**Recon recommendation:** the spike should use option (A) — caller-asserted via a fixture that knows its inputs are sorted. Production wiring decision (B vs C vs D) is iteration-1 plan territory, not spike territory. The spike's job is to answer "is sort-merge ever fast enough to be worth dispatching", not "how should we dispatch it".

### 4. Existing dead-code surface (unwired)

* `crates/xlog-runtime/src/statistics.rs:15` — `JoinStrategy::SortMerge` enum variant exists. Production-unwired (same status W4.2 left the `NestedLoop` variant in). Per W4.2 D8 process locks pattern, W4.3 should similarly leave the existing dead enum untouched and introduce its own threshold/dispatch surface.

## Value-claim spike question

> "Does sort-merge join (with caller-asserted pre-sorted inputs) beat `hash_join_v2` in a useful region of input sizes?"

**Useful region:** any input-size range where the speedup ≥ 2× (matching W4.2's production acceptance bar). If the spike finds no such region, the operator can still ship for *correctness* (some upstream consumer might want it) but production dispatch wiring would be unjustified.

### Why a spike is necessary (not skippable)

1. **Hash's per-launch overhead floor** (~700µs–2.7ms per W4.2 spike F2) is the dominant cost at small sizes. A new sort-merge kernel may have its own launch overhead that nullifies the "no hash-table build" advantage. Need empirical data.

2. **Sort-merge's ALGORITHMIC win is asymptotic**: O(L + R) merge pass vs hash's O(L + R) probe. Both linear. The constant-factor difference depends on kernel implementation, memory access patterns, divergence — measurable, not derivable.

3. **Production multi-col + payload**: same caveat F-W42-3 surfaced for nested-loop applies — spike's 1-col kernel measures kernel-side cost; production multi-col needs the gather pass, which compresses the speedup. The spike establishes upper-bound; production bench validates final acceptance.

### Spike proposal (locked scope)

Mirroring the W4.2 spike-first discipline:

| # | Lock | Proposed value |
|---|------|----------------|
| 1 | Worktree + branch (unmerged regardless of outcome) | `.worktrees/w43-bench-spike-sort-merge` on `bench-spike/w43-sort-merge` (NOT `feat/w43-sort-merge-join`) |
| 2 | Kernel scope (minimum viable) | Inner join only, U32 only, 1-key, 1-col arity. Caller asserts pre-sorted. Same minimum-viable shape that W4.2's spike used. |
| 3 | Bench envelope parity | Provider-direct `provider.sort_merge_join_v2_inner_u32_1key_1col_spike(...)` vs `provider.hash_join_v2(...)` on the same uploaded `CudaBuffer` inputs. |
| 4 | Row-set parity check | `BTreeSet<u32>` equality on every measured fixture (sorted-unique inputs make matched-key sets unambiguous). |
| 5 | Falsification matrix | Symmetric `(N, N)` and asymmetric `(N, M)` cells across 50–10000 row-counts. Match the W4.2 spike's matrix for cross-comparison. |
| 6 | Decision gate after spike | If sort-merge wins by ≥ 2× in any useful region → iteration-1 plan locks the dispatch threshold from spike data + runtime-detection mechanism choice. If hash always wins or speedup < 2× → spike preserved as evidence; W4.3 re-scoped (operator-only without dispatch wiring, OR closed as "not worth it"). |

## Open questions for iteration-1 plan (out of spike scope)

1. **Production sortedness detection mechanism** — pick A (caller flag), B (runtime detection kernel a-la W3.2), C (buffer metadata), or D (IR annotation).
2. **Dispatch precedence** — sort-merge vs nested-loop (W4.2) vs hash. If a join is eligible for both nested-loop AND sort-merge, which wins? The answer depends on spike data.
3. **Threshold semantic** — Cartesian product (per W4.2) or row-count-based? Sort-merge's algorithmic profile differs from nested-loop's, so a different threshold may be appropriate.
4. **Schema/key-type admissibility** — same Inner + 1-key + U32/Symbol envelope as W4.2, or wider?
5. **Hash-fallback policy on detection failure** — if runtime detection (option B) decides "not sorted", do we fall back to hash, or do we sort-then-merge?

These are NOT spike questions; they're plan-iteration questions to be answered AFTER the spike data lands.

## Recommendation

**Proceed with spike** before drafting any iteration-1 plan content. Spike scope as locked above. Decision gate determines whether iteration-1 is worth drafting at all.

Awaiting your authorization for the spike.
