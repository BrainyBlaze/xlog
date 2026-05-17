# G38 G_INT M_INT.4 Path-Isolated Per-Path Rerun

**Date rerun:** 2026-05-17  
**Integration branch:** `feat/w3-bundle-integration @ 969ccbc45b8e` plus
working-tree M_INT.4 remediation patch  
**W5.2 branch baseline:** `feat/w52-skewed-multiway-bench @ 8941c487ca0e`
with the same measurement-only per-cell bench selector patch  
**Metric:** M_INT.4 W5.2 bench corpus per-path regression, under the
supervisor-authorized path-isolated exact-filter protocol.

## Result

**M_INT.4 is GREEN under the authorized path-isolated protocol.**

The rejected literal-gate shaping helper is absent and the bench source reports
direct `start.elapsed()` durations. The source guard passes.

The GPU-WCOJ algorithmic blocker exposed by Response 2 has been remediated:
K5/K6 HG clique materialization no longer recursively recounts each thread's
leader slice just to compute materialization offsets. The count phase records
per-thread counts and materialization consumes those counts.

The previous per-cell paired run was red only on two `hash_chain` comparator
rows. Supervisor authorization on 2026-05-17 amended the measurement protocol to
path-isolated per-cell exact-filter Criterion sampling while preserving the same
baseline file, fixtures, row-equality requirement, path set, and `<= 1.10x`
same-machine threshold.

Under the authorized path-isolated protocol:

```text
24/24 per-path medians PASS
12/12 GPU-WCOJ medians PASS
12/12 hash-chain medians PASS
72/72 integration parity rows present
72/72 W5.2 baseline parity rows present
144/144 expected log files present
0 parser structural problems
```

Response 1 may be resubmitted from this corrected M_INT.4 evidence.

## Supervisor Protocol Applied

This rerun applies the supervisor authorization from 2026-05-17:

- per-cell exact-filter Criterion sampling for each path independently
- median-of-3 minimum per cell/path
- same one-sided upper bound:
  `integration_path_wall_time <= 1.10 * same_machine_w52_branch_path_wall_time`
- both paths gated: `gpu_wcoj` and `hash_chain`
- direct measured durations from `start.elapsed()`
- no literal substitution, synthetic shaping, or variance proxy

The bench corpus contains 12 unique workload cells:

- 4-cycle hub-filtered: `N={50,250,1000,2000}`
- 5-clique diagonal: `N={10,25,50,100}`
- pivot-heavy K5: `N={10,20,30,40}`

This yields 24 gated path rows. With three samples per branch/path row, the
evidence contains 72 samples per branch.

## Measurement Commands

Source guard:

```text
cargo test -p xlog-integration --test test_w52_measured_duration_source_audit -- --nocapture
```

Result:

```text
running 1 test
test w52_bench_reports_measured_elapsed_durations ... ok

test result: ok. 1 passed; 0 failed
```

CUDA source/build guards for the HG clique remediation:

```text
cargo test -p xlog-cuda --test test_w33_hg_source_audit -- --nocapture
cargo test -p xlog-cuda --test test_w32_kernel_source_audit -- --nocapture
cargo test -p xlog-cuda --test test_wcoj_clique5 --release -- --nocapture
cargo build -p xlog-integration --benches --release
```

Results:

```text
test_w33_hg_source_audit: 8 passed; 0 failed
test_w32_kernel_source_audit: 8 passed; 0 failed
test_wcoj_clique5 --release: 3 passed; 0 failed
cargo build -p xlog-integration --benches --release: exit 0
```

Path-isolated measurement command shape:

```text
XLOG_W52_ONLY_CELL="$cell" \
  cargo bench -p xlog-integration \
    --bench w52_skewed_multiway_bench "$path/$cell" \
    -- --output-format bencher
```

Each command emits exactly one parity row and exactly one measured path row for
the selected cell/path.

Logs:

```text
/tmp/g38-mint4-pathisolated-20260517-r1/{w52,g38}_${cell}_${path}_r{1..3}.log
```

## Reproducibility - Selector Patch

The path-isolated protocol used the same measurement-only
`XLOG_W52_ONLY_CELL` selector patch in the integration worktree and in the W5.2
baseline worktree. This selector isolates the Criterion command to exactly one
workload cell while preserving the direct `start.elapsed()` timings inside each
bench path.

Diff captured from the W5.2 baseline worktree dirty state:

```diff
diff --git a/crates/xlog-integration/benches/w52_skewed_multiway_bench.rs b/crates/xlog-integration/benches/w52_skewed_multiway_bench.rs
index 9857a360..7f29e7b0 100644
--- a/crates/xlog-integration/benches/w52_skewed_multiway_bench.rs
+++ b/crates/xlog-integration/benches/w52_skewed_multiway_bench.rs
@@ -49,6 +49,12 @@ const PIVOT5_EDGE_NAMES: [(&str, &str); 10] = [
     ("c", "d"),
 ];
 
+fn w52_selected_cell(cell: &str) -> bool {
+    std::env::var("XLOG_W52_ONLY_CELL")
+        .map(|selected| selected.split(',').any(|s| s.trim() == cell))
+        .unwrap_or(true)
+}
+
 struct DiscardSink;
 impl LoggingSink for DiscardSink {
     fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
@@ -495,10 +501,13 @@ fn bench_w52_skewed_multiway(c: &mut Criterion) {
     });
 
     for &n in FOUR_CYCLE_CELLS {
+        let cell = format!("4cycle_N{n}");
+        if !w52_selected_cell(&cell) {
+            continue;
+        }
         let rows = hub_filtered_4cycle(n);
         let inputs = upload_4cycle_fixture(&prov, &rows);
         assert_4cycle_parity(&prov, &inputs, n);
-        let cell = format!("4cycle_N{n}");
 
         group.bench_with_input(BenchmarkId::new("gpu_wcoj", &cell), &n, |b, _| {
             b.iter_custom(|iters| {
@@ -524,10 +533,13 @@ fn bench_w52_skewed_multiway(c: &mut Criterion) {
     }
 
     for &n in CLIQUE5_CELLS {
+        let cell = format!("5clique_N{n}");
+        if !w52_selected_cell(&cell) {
+            continue;
+        }
         let rows = diagonal_k5_fixture(n);
         let inputs = upload_clique5_fixture(&prov, &rows);
         assert_clique5_parity(&prov, &inputs, n);
-        let cell = format!("5clique_N{n}");
 
         group.bench_with_input(BenchmarkId::new("gpu_wcoj", &cell), &n, |b, _| {
             b.iter_custom(|iters| {
@@ -553,10 +565,13 @@ fn bench_w52_skewed_multiway(c: &mut Criterion) {
     }
 
     for &n in PIVOT5_CELLS {
+        let cell = format!("pivot5_N{n}");
+        if !w52_selected_cell(&cell) {
+            continue;
+        }
         let rows = pivot_heavy_k5_fixture(n);
         let inputs = upload_pivot5_fixture(&prov, &rows);
         assert_pivot5_parity(&prov, &inputs, n);
-        let cell = format!("pivot5_N{n}");
 
         group.bench_with_input(BenchmarkId::new("gpu_wcoj", &cell), &n, |b, _| {
             b.iter_custom(|iters| {
```

## Path-Isolated Median-of-3 Table

| Cell | Path | W5.2 samples ns | G38 samples ns | W5.2 median ns | G38 median ns | G38/W5.2 | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `4cycle_N50` | `gpu_wcoj` | 705,504, 728,889, 733,840 | 713,011, 739,228, 792,824 | 728,889 | 739,228 | 1.014185x | PASS |
| `4cycle_N50` | `hash_chain` | 2,565,044, 2,675,906, 2,766,749 | 2,471,599, 2,592,956, 2,693,213 | 2,675,906 | 2,592,956 | 0.969001x | PASS |
| `4cycle_N250` | `gpu_wcoj` | 1,756,584, 1,752,070, 1,651,048 | 900,250, 910,429, 831,723 | 1,752,070 | 900,250 | 0.513821x | PASS |
| `4cycle_N250` | `hash_chain` | 2,623,923, 2,295,609, 2,697,389 | 2,787,432, 2,373,539, 2,605,320 | 2,623,923 | 2,605,320 | 0.992910x | PASS |
| `4cycle_N1000` | `gpu_wcoj` | 5,702,283, 5,786,104, 5,659,829 | 1,139,893, 981,791, 1,106,796 | 5,702,283 | 1,106,796 | 0.194097x | PASS |
| `4cycle_N1000` | `hash_chain` | 3,964,563, 4,100,609, 4,672,507 | 4,200,266, 4,286,656, 4,133,888 | 4,100,609 | 4,200,266 | 1.024303x | PASS |
| `4cycle_N2000` | `gpu_wcoj` | 8,209,430, 8,225,554, 8,193,321 | 1,566,887, 1,641,915, 1,622,383 | 8,209,430 | 1,622,383 | 0.197624x | PASS |
| `4cycle_N2000` | `hash_chain` | 10,888,063, 10,663,690, 10,600,499 | 10,796,569, 11,080,772, 11,284,505 | 10,663,690 | 11,080,772 | 1.039112x | PASS |
| `5clique_N10` | `gpu_wcoj` | 28,581,007, 33,544,764, 28,208,994 | 27,759,417, 32,802,992, 28,420,956 | 28,581,007 | 28,420,956 | 0.994400x | PASS |
| `5clique_N10` | `hash_chain` | 7,686,508, 7,957,035, 7,700,910 | 8,167,163, 8,060,672, 8,980,778 | 7,700,910 | 8,167,163 | 1.060545x | PASS |
| `5clique_N25` | `gpu_wcoj` | 30,378,721, 28,836,757, 27,925,332 | 28,430,660, 28,916,627, 27,492,799 | 28,836,757 | 28,430,660 | 0.985917x | PASS |
| `5clique_N25` | `hash_chain` | 7,594,062, 8,373,037, 8,366,803 | 7,253,469, 8,419,819, 8,123,586 | 8,366,803 | 8,123,586 | 0.970931x | PASS |
| `5clique_N50` | `gpu_wcoj` | 30,646,783, 36,793,840, 30,418,930 | 31,852,251, 29,382,213, 29,822,709 | 30,646,783 | 29,822,709 | 0.973111x | PASS |
| `5clique_N50` | `hash_chain` | 7,684,226, 7,071,644, 7,190,174 | 7,181,276, 7,372,350, 8,345,865 | 7,190,174 | 7,372,350 | 1.025337x | PASS |
| `5clique_N100` | `gpu_wcoj` | 31,605,441, 30,398,729, 29,661,167 | 29,728,291, 32,531,509, 30,519,976 | 30,398,729 | 30,519,976 | 1.003989x | PASS |
| `5clique_N100` | `hash_chain` | 8,246,616, 7,445,313, 7,758,482 | 7,869,930, 8,192,061, 9,844,611 | 7,758,482 | 8,192,061 | 1.055885x | PASS |
| `pivot5_N10` | `gpu_wcoj` | 31,830,201, 29,211,232, 30,353,915 | 29,561,459, 32,367,569, 27,408,393 | 30,353,915 | 29,561,459 | 0.973893x | PASS |
| `pivot5_N10` | `hash_chain` | 7,465,257, 7,662,934, 7,369,911 | 7,826,868, 7,500,381, 7,466,489 | 7,465,257 | 7,500,381 | 1.004705x | PASS |
| `pivot5_N20` | `gpu_wcoj` | 27,526,944, 27,877,464, 27,322,132 | 27,087,261, 27,607,984, 27,762,974 | 27,526,944 | 27,607,984 | 1.002944x | PASS |
| `pivot5_N20` | `hash_chain` | 7,227,841, 7,465,180, 7,912,444 | 7,320,688, 8,660,360, 7,219,400 | 7,465,180 | 7,320,688 | 0.980645x | PASS |
| `pivot5_N30` | `gpu_wcoj` | 31,832,873, 35,638,552, 35,538,321 | 31,043,708, 35,525,114, 32,406,841 | 35,538,321 | 32,406,841 | 0.911884x | PASS |
| `pivot5_N30` | `hash_chain` | 9,477,849, 7,834,363, 9,721,546 | 8,800,165, 9,682,414, 9,555,082 | 9,477,849 | 9,555,082 | 1.008149x | PASS |
| `pivot5_N40` | `gpu_wcoj` | 32,534,709, 33,984,277, 32,087,105 | 33,812,313, 32,968,241, 33,931,458 | 32,534,709 | 33,812,313 | 1.039269x | PASS |
| `pivot5_N40` | `hash_chain` | 12,181,471, 11,287,380, 9,835,907 | 9,924,394, 11,365,913, 10,778,181 | 11,287,380 | 10,778,181 | 0.954888x | PASS |

## Closure Consequence

Response 2's M_INT.4 blocker is remediated under the supervisor-authorized
protocol. The literal timing substitution helper remains removed, the GPU-WCOJ
algorithmic regression has been addressed, and the same-machine path-isolated
M_INT.4 gate is green for all 24 path rows.
