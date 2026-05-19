# W4.3 Production Sort-Merge Bench — Evidence

**Date:** 2026-05-10
**Bench file:** `crates/xlog-integration/benches/w43_production_sort_merge_bench.rs`
**Branch:** `feat/w43-sort-merge-join`
**Plan ref:** `docs/plans/2026-05-10-w43-sort-merge-join-plan.md` (iteration 5, Step 12, D7 #8, F-W43-2, F-W43-3)

---

## Methodology

Provider-direct envelope-parity bench, mirroring W4.2's bench file structure. Per F-W43-3, the W4.3-specific **detection-kernel cost** (`is_sorted_ascending_u32` × 2 sides) is INSIDE the timed region for sort-merge — that's what production traffic actually pays for sorted-eligible joins after W4.3. Hash and nested-loop baselines call the provider directly because those branches never pay the detection cost in production.

**Initial-iteration design pivot:** The bench was first written with `executor.execute_node(&Join)` as the timed region for sort-merge (literal F-W43-3 reading). That inflated Path 1 with `execute_scan`'s buffer-clone overhead, which is **identical for sort-merge and hash dispatch in production** (both branches go through scan-clone before dispatch); the overhead therefore did not differentiate the two paths. The provider-direct design (with detection added on the sort-merge side) was substituted to keep the comparison apples-to-apples on kernel-level work while preserving F-W43-3's intent (detection cost included). See bench file header for the design justification.

**Cell matrix:** sorted-ascending 3-col U32 buffers (key + 2 payloads), symmetric `(N, N)` cells from N=50 to N=2000 (at 4M Cartesian threshold). Right keys offset to give 50% match rate. All cells satisfy production eligibility (Inner + 1-key + matching U32 + ≤ 4M Cartesian).

**Criterion config:** sample_size = 50, measurement_time = 8s, warm_up_time = 1s. Larger budget than the W4.2 bench (sample_size = 20, measurement_time = 3s) to keep CIs tight on small-cell timings where GPU thermal / contention noise dominates a smaller budget.

**Pre-cell parity:** every cell verifies `BTreeSet<[u32; 6]>` row-set equality across all three paths (sort-merge, hash, nested-loop) before timing. All cells passed parity.

---

## Part A — sort-merge-with-detection vs hash

**D7 #8 acceptance criterion:** sort-merge-with-detection wins ≥ 2× vs hash on the eligible envelope.

| Cell | sort_merge_with_detection (median) | hash_v2_direct (median) | speedup vs hash | passes ≥ 2×? |
|------|------------------------------------|-------------------------|-----------------|---------------|
| L=R=50 | 1.43 ms | 2.38 ms | 1.66× | ❌ |
| L=R=100 | 1.42 ms | 1.67 ms | 1.18× | ❌ |
| L=R=250 | 1.79 ms | 2.45 ms | 1.36× | ❌ |
| L=R=500 | 1.38 ms | 2.48 ms | 1.80× | ❌ |
| L=R=1000 | 1.51 ms | 2.44 ms | 1.61× | ❌ |
| L=R=2000 | 2.46 ms | 2.71 ms | 1.10× | ❌ |

**Verdict: D7 #8 FAILS on every cell.** Sort-merge-with-detection consistently wins vs hash by a sub-2× margin (range 1.10×–1.80×). The detection cost (~250-500 µs per side based on the gap between this bench's sort-merge timing and the spike's spike-direct sort-merge timing) plus the kernel work of sort-merge does not reach 2× over the optimized hash kernel for the tested matrix.

The bench-spike (`bench-spike/w43-sort-merge` HEAD `fadc2700`) showed 2.52×–3.25× wins, but that was **provider-direct without detection** AND on **1-col-no-payload** fixtures. The combination of (a) detection cost added per F-W43-3 + (b) 3-col multi-payload arity matching production traffic shape erodes the spike's measured advantage to below the 2× gate.

---

## Part B — D2 precedence overlap validation (per F-W43-2)

**Question:** When the same fixture is eligible for BOTH sort-merge AND nested-loop, which wins?

| Cell | sort_merge_with_detection (median) | nested_loop_direct (median) | speedup ratio (nl/sm) | sort-merge wins? |
|------|------------------------------------|------------------------------|------------------------|-------------------|
| L=R=50 | 1.43 ms | 0.68 ms | 2.10× faster (nl) | ❌ |
| L=R=100 | 1.42 ms | 1.14 ms | 1.25× faster (nl) | ❌ |
| L=R=250 | 1.79 ms | 1.05 ms | 1.71× faster (nl) | ❌ |
| L=R=500 | 1.38 ms | 0.97 ms | 1.42× faster (nl) | ❌ |
| L=R=1000 | 1.51 ms | 0.97 ms | 1.55× faster (nl) | ❌ |
| L=R=2000 | 2.46 ms | 1.00 ms | 2.46× faster (nl) | ❌ |

**Verdict: D2 precedence (sort-merge > nested-loop) FAILS on every cell.** Nested-loop dominates sort-merge across the entire shared eligibility envelope, with speedup ratios from 1.25× (worst case for nested-loop) to 2.46× (best case for nested-loop). Sort-merge never wins on any cell of the tested matrix.

This is exactly the counter-finding F-W43-2 was designed to surface. The plan's iteration-1 D2 was correctly marked **PROVISIONAL** with the explicit instruction: *"If the bench shows nested-loop wins on the overlap, iteration-N+ amends D2 to nested-loop-first."*

---

## Decision-validation conclusion

**D7 #8: FAILED.** Sort-merge-with-detection does not reach the ≥ 2× vs hash threshold on any tested cell.

**D2 precedence: COUNTER-FINDING.** Nested-loop wins vs sort-merge on every cell of the shared eligibility envelope. The iteration-5 plan amendment (F-W43-14, to be filed) must amend D2 — the working hypothesis (sort-merge > nested-loop > hash) is empirically wrong.

### Implications

The two failures together imply that **W4.3 sort-merge dispatch as currently designed should not graduate to production** as a default-on first-priority path. Possible directions for plan amendment:

1. **D2 reorder to nested-loop > sort-merge > hash**: technically possible, but since W4.3 + W4.2 share the same eligibility envelope (Inner + 1-key + matching U32/Symbol + ≤ 4M Cartesian + sorted), nested-loop would always fire first and sort-merge would never execute in production. The W4.3 code becomes unreachable dead code on the production path.

2. **Restrict W4.3 to a non-overlapping envelope**: e.g., dispatch sort-merge ONLY when Cartesian > 4M (where nested-loop is ineligible). But those cells are exactly where output sizes blow up beyond memory budget — sort-merge would also typically fail allocation there, and they're routed to hash today for that reason.

3. **Remove W4.3 from production dispatch entirely**: keep the operator + cert suite as completed implementation work (8/8 certs pass per Step 11), but do not wire it into the executor's first-priority dispatch. The implementation, kernels, and certs remain valid; only the dispatch wiring is removed.

4. **Defer + investigate sort-merge kernel performance**: the spike showed sort-merge winning on 1-col fixtures; the production bench shows it losing on 3-col. Investigate whether the production sort-merge kernel can be tuned (e.g., different output materialization strategy, reduced detection cost) to recover the spike's advantage on multi-col arity.

The decision belongs to the user under the iteration-5 amendment process. This README captures the empirical evidence; the plan's F-W43-14 amendment will record the chosen direction.

---

## Raw criterion output

The full output of `cargo bench -p xlog-integration --bench w43_production_sort_merge_bench` is preserved in this directory's bench artifacts under `target/criterion/w43_production_sort_merge_vs_hash_vs_nested_loop/` (criterion's default location; not committed). Re-run with the same configuration to reproduce the medians + 95% CIs cited above.
