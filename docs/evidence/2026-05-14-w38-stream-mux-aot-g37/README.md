# W3.8 G5/S5.2-S5.4 Stream-Mux AOT Production Evidence

Status: PASS production branch for G5/W3.8.

Branch: `feat/w38-stream-mux-aot-g37`
Base: `feat/w37-helper-split-aot-g37 @ bfd80d67`
Passing spike: `bench-spike/w38-stream-mux-recompute-g37 @ c0644eec`
Bench command: `cargo bench -p xlog-integration --bench wcoj_w38_stream_mux -- --output-format bencher`
Compile gate: `cargo bench -p xlog-integration --bench wcoj_w38_stream_mux --no-run`

## Production Changes

- Added `optimizer::stream_schedule_pass::schedule_streams`, which builds the Count -> Scan -> Resize -> Materialize phase order for independent WCOJ rules.
- Added `HardwareCapabilities`, `StreamPhase`, `StreamPhaseNode`, and `StreamSchedule` as the AOT schedule model.
- Added uncapped triangle U32 split phase hooks:
  - Count phase uses `wcoj_triangle_count_hg_u32`.
  - Materialize phase uses `wcoj_triangle_materialize_hg_u32` and recomputes from deterministic offsets.
- Added `wcoj_w38_stream_mux` production benchmark for M5.1-M5.4.

The production phase surface intentionally does not use `wcoj_triangle_count_hg_cached_u32`, because the RED spike proved its fixed local scratch capacity drops rows at wider block work units.

## Rule-Rich Cell

Fixture: four independent superhub triangle rules, `ROWS=2,000,000`, `BLOCK_WORK_UNIT=65,536`.

| Metric | Result | Verdict |
|---|---:|---|
| M5.3 row equality | PASS for all 4 rules | PASS |
| Rule 0 rows / total_work | 1,205,334 / 4,116,962 | PASS |
| Rule 1 rows / total_work | 1,204,608 / 4,119,201 | PASS |
| Rule 2 rows / total_work | 1,204,079 / 4,115,828 | PASS |
| Rule 3 rows / total_work | 1,204,075 / 4,115,329 | PASS |
| Direct sequential wall | 86,201.401 us | info |
| Direct mux wall | 38,816.850 us | info |
| Direct speedup | 2.221x | PASS M5.1 |
| Direct concurrency, `sum_single / mux_wall` | 2.261x | PASS M5.4 |
| Criterion sequential median | 17,108,994 ns/iter | info |
| Criterion mux median | 8,013,383 ns/iter | info |
| Criterion speedup | 2.135x | PASS M5.1 |

M5.1 requires at least 1.27x speedup. M5.4 requires at least 1.2x concurrency.

## Single-Rule Cell

Fixture: one superhub triangle rule, `ROWS=2,000,000`, `BLOCK_WORK_UNIT=65,536`.

| Metric | Result | Verdict |
|---|---:|---|
| Direct sequential wall | 22,065.236 us | info |
| Direct scheduled wall | 21,327.022 us | PASS |
| Direct ratio, sequential / scheduled | 1.035x | PASS M5.2 |
| Criterion sequential median | 4,223,750 ns/iter | info |
| Criterion scheduled median | 4,246,349 ns/iter | info |
| Criterion scheduler overhead | +0.535% | PASS M5.2 |

M5.2 permits no regression beyond +3% on single-rule strata. The scheduler path is within +0.535%.

## Targeted Tests

`cargo test -p xlog-logic --lib stream_schedule_pass -- --nocapture`

Result: 2 passed, 0 failed.

Covered checks:

- Four independent rules on a 16-SM hardware input schedule onto four stream slots.
- Phase nodes are grouped Count, Scan, Resize, Materialize.
- Single-rule stratum schedules onto one stream.

## Verdict

G5/W3.8 production metrics M5.1-M5.4 pass on this branch. The stream-mux production path should be integrated after review into the bundle integration branch, then rechecked by G6's bundle-scale benchmark harness.
