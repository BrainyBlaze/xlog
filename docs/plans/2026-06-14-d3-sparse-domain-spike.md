# D3 Sparse-Domain Factorized Delta — Bench Spike

Date: 2026-06-14. Branch `feat/d3-sparse-domain` (worktree), from main `8da79e8e`
(D3 Phase B merged). Precondition: S3+S4 dense gates PASS.

## 1. Why

Phase B's factorized delta wins only in the **dense-domain** regime: the novel set is
evaluated over a characteristic bitvector of `domain²/8` bytes, so it is gated to
`domain ≤ 2¹⁶` and declines (legacy hash-join → diff) for everything else. Real recursive
workloads — transitive closure over large sparse graphs (node ids spanning millions, low
average degree) — fall outside that cap and never benefit. This spike asks the open
question from the research report (§4 D3, §6.1): **is there a factorized novel-set
evaluation that wins on a sparse, large-domain delta-blowup workload, where a dense
bitvector is infeasible?**

## 2. The idea under test

The legacy semi-naive step materializes every derivation **witness** of the join
`delta(x,y) ⋈ edge(y,z)` — one flat row per (x,y,z) — then sorts that buffer, dedups to
distinct (x,z), and diffs against the stable relation R. On a delta-blowup workload the
witness count is `≫` the distinct-novel count (the C1/C3 blowup), so the materialized
intermediate and its sort dominate peak memory and time.

The factorized sparse step keeps the candidates as a **d-representation** (per delta row:
x × the edge[y] trie range — never expanded) and evaluates the novel set with a single
GPU **open-addressing hash set** keyed by the 64-bit composite `(x<<32)|z`:

1. Pre-load R into the hash set (mark its slots "known").
2. Grid-stride over candidate work items `(delta row i, range offset k)`; for each,
   compute `(x,z)`, `atomicCAS`-insert into the set **only if absent and not-known** —
   duplicate witnesses and rediscoveries collapse at the slot with no materialization.
3. Compact the newly-inserted slots into the output (x,z) buffer.

Peak ≈ `|R| + table` (table sized to `≈2× (|R| + expected novel)`), independent of witness
multiplicity. No `domain²` term, so it scales to sparse large-domain graphs. The win vs
legacy is the eliminated witness-materialized sort buffer (C1) and the eliminated
re-sort-of-R-every-iteration inside `diff_gpu` (C3) — exactly the dense spike's win, via a
hash set instead of a bitvector.

Output is **unordered** (hash-set scan), unlike the dense path's lex-sorted emit, so the
sparse novel set is dedup-by-construction but must be sorted if a caller needs order
(union_gpu already sorts, so the fixpoint is unaffected).

## 3. Spike scope (minimum viable)

- One CUDA kernel family in a new `kernels/fj_delta_sparse.cu` (or appended to wcoj.cu):
  `fj_delta_sparse_insert_u32` (load-R + candidate-insert, two entry modes or two
  kernels) over a `u64` open-addressing table of `u32` slot payloads (slot stores a packed
  index or the (x,z) directly); plus a compaction pass (mask scanned-inserted slots →
  emit). Reuse the existing scan/compaction kernels.
- Provider entry `fj_delta_sparse_novel_u32_recorded(delta, edge_norm, full_r, cols, stream)`
  in `provider/fj_delta_sparse.rs` — same signature shape as the dense entry minus
  `domain` (no cap), plus a table-size policy (load factor ≤ 0.5, fail-closed if the table
  would exceed a budget fraction).
- NOT wired into the executor dispatch yet (spike only). NOT u64-key-width, NOT
  multi-predicate. Pure standalone spike measured against the gate.

## 4. Gate (per perf discipline; fail → park branch unmerged with evidence)

Fixture: sparse large-domain delta-blowup. A random-ish sparse digraph with controlled
witness multiplicity — e.g. `n = 2²⁰` nodes (domain far over the dense 2¹⁶ cap), average
out-degree d ≈ 4, structured so each novel TC pair has many witnesses (layered/blocked so
distinct-novel ≪ witnesses). Deterministic construction (no RNG in-test; seed by index).

| metric | gate |
|---|---|
| peak memory | factorized < legacy (target ≥2× on the blowup fixture) |
| wall-clock | ≤1.2× legacy |
| correctness | deterministic row-set parity vs CPU oracle AND vs legacy engine |

Measurement: RunPod minimal A4000, `--startSSH`, `XLOG_CUBIN_ARCHS=sm_86`; interleaved
A/B per rep + warm-up (the S4 methodology fix); 3 reps median; clocks/temp logged; pod
deleted + confirmed.

## 5. Risks / kill criteria

- **Hash collisions / load factor**: open addressing degrades near full; size to ≤0.5 and
  fail-closed if the upper-bound (|R| + total candidate work) exceeds budget — that bound
  is loose (candidates ≫ novel), so the table may be over-sized vs the dense bitvector.
  If the table itself blows peak past legacy, the spike FAILS — that is a real finding.
- **atomicCAS contention** on hot keys (high-multiplicity novel pairs hammer one slot):
  this is the sparse analogue of the dense `atomicOr` contention; if it dominates, the
  spike fails the wall-clock bar.
- **R-membership**: pre-loading all of R into the table each iteration is `O(|R|)` work
  per iteration (like the dense subtract). Acceptable iff the witness saving dominates.
- If neither peak nor wall-clock beats legacy on a genuinely sparse blowup, **park**: the
  dense bitvector remains the only factorized win and sparse stays on the legacy path
  (the honest Phase B boundary). No production wiring without a PASS.

## 6. Evidence — PASS (2026-06-14)

RunPod RTX A4000, HEAD `1ff552dc`, evidence `docs/evidence/2026-06-14-sparse-domain-spike/`.
Hub-blowup fixture (512 sources × 64 hubs × 512 sink = 16.78M witnesses → 262,144 distinct
novel over a ~2²² domain; dense bitvector would need 2⁴⁴/8 bytes — infeasible):

| | wall-clock | peak | rows |
|---|---|---|---|
| sparse hash-set | **6.7 ms** | **418.8 MiB** | 262,144 |
| legacy hash_join+project+dedup | 123.6 ms | 836.8 MiB | 262,144 |
| ratio | **0.054×** (18× faster) | **2.00× reduction** | parity ✓ |

PASS on both gate bars (peak < legacy at target ≥2×; wall-clock ≤1.2×). The over-provisioned
table (2²⁵ slots ≈ 428 MiB) still beats legacy 2× on peak because legacy materializes the
16.78M witness rows **plus** the join hash table **plus** dedup sort scratch (together
> the over-sized table), and the hash set does no sort. The earlier worry that
over-provisioning would lose on peak was refuted by measurement.

Local parity: large-id (2²⁰, over the dense 2¹⁶ cap), empty/saturated, and sparse-vs-dense
cross-parity all green.

**Scope / honesty**: this is a single-step spike (R empty, one semi-naive iteration) measured
in isolation — the sparse analogue of the dense S3 spike, NOT production integration. The
table is sized to a conservative `2×(|R|+candidate work)` upper bound with a fail-closed
guard; a production sparse path would want distinct-count-aware sizing (2-pass or growth) so
the table does not exceed budget on workloads where `total_work` is enormous. **Decision**:
spike PASS → sparse factorization is viable; branch preserved unmerged as spike evidence.
Next phase (separate plan): production integration — a domain-based router selecting
dense-bitvector vs sparse-hash-set vs legacy inside `try_dispatch_factorized_delta`, then a
full-fixpoint S4-equivalent bench before any merge.
