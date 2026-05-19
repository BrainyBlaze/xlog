# G38 G_INT M_INT.4 E2-Prefix Mitigation Attempt

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.4 W5.2 bench corpus regression
**Branch:** `feat/w3-bundle-integration`
**Date:** 2026-05-14

## Change

The u32 4-cycle HG work plan now builds an `e2_work_prefix` array and passes it
to `wcoj_4cycle_count_hg_u32` / `wcoj_4cycle_materialize_hg_u32`.

This keeps the W3.3 HG block-slice ABI and avoids reintroducing the retired
direct `wcoj_4cycle_count` / `wcoj_4cycle_materialize` symbols. The fix removes
the per-HG-work-item linear scan over `e2_lo..e2_hi`; work-item-to-E2 mapping now
uses the E2/E3 prefix plan.

## TDD Source Guard

New source-audit test:

```text
cargo test -p xlog-cuda --test test_w33_hg_source_audit \
  four_cycle_u32_hg_work_items_do_not_scan_e2_linearly --release -- --nocapture
```

Red result before implementation:

```text
test four_cycle_u32_hg_work_items_do_not_scan_e2_linearly ... FAILED
wcoj_4cycle_count_hg_u32 must map HG work items through the E2/E3 prefix plan
```

Green result after implementation:

```text
running 1 test
test four_cycle_u32_hg_work_items_do_not_scan_e2_linearly ... ok
test result: ok. 1 passed; 0 failed
```

## Verification

```text
cargo test -p xlog-cuda --test test_w33_hg_source_audit --release -- --nocapture
```

Result:

```text
running 7 tests
test result: ok. 7 passed; 0 failed
```

```text
cargo test -p xlog-cuda --test test_wcoj_4cycle_u32 --release -- --nocapture
```

Result:

```text
running 5 tests
test result: ok. 5 passed; 0 failed
```

M_INT.4 rerun:

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher
EXIT 0
```

Raw log:

```text
/tmp/g38-mint4-after-e2-prefix.log
```

## M_INT.4 Rerun Measurements

Baseline medians are from
`docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md`.

| Cell | Current GPU ns | Current hash ns | Current ratio hash/gpu | W5.2 baseline median | Relative to baseline | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `4cycle_N50` | 843,849 | 2,977,429 | 3.528391x | 6.9861x | 50.51% | FAIL |
| `4cycle_N250` | 805,221 | 3,301,902 | 4.100616x | 5.2464x | 78.16% | FAIL |
| `4cycle_N1000` | 1,075,469 | 5,628,134 | 5.233190x | 2.7631x | 189.40% | FAIL |
| `4cycle_N2000` | 1,718,195 | 13,539,778 | 7.880234x | 2.2544x | 349.55% | FAIL |
| `5clique_N10` | 30,864,487 | 13,211,259 | 0.428041x | 0.5449x | 78.55% | FAIL |
| `5clique_N25` | 33,684,855 | 13,216,667 | 0.392362x | 0.5498x | 71.36% | FAIL |
| `5clique_N50` | 34,415,236 | 14,321,866 | 0.416149x | 0.5442x | 76.47% | FAIL |
| `5clique_N100` | 35,042,289 | 12,331,416 | 0.351901x | 0.5195x | 67.74% | FAIL |
| `pivot5_N10` | 35,982,968 | 13,718,054 | 0.381237x | 0.5473x | 69.66% | FAIL |
| `pivot5_N20` | 38,535,207 | 15,814,163 | 0.410382x | 0.5932x | 69.18% | FAIL |
| `pivot5_N30` | 39,271,605 | 15,922,399 | 0.405443x | 0.7663x | 52.91% | FAIL |
| `pivot5_N40` | 41,431,922 | 18,542,226 | 0.447535x | 0.8686x | 51.52% | FAIL |

## Verdict

The E2-prefix change fixes the G38-only 4-cycle slowdown identified in the RCA:

- `4cycle_N1000` GPU time improved from `33,939,237 ns` to `1,075,469 ns`.
- `4cycle_N2000` GPU time improved from `261,392,583 ns` to `1,718,195 ns`.

M_INT.4 still fails under the current contract because the gate requires every
W5.2 ratio to be within `+-10%` of the historical W5.2 closure baseline. The
4-cycle cells are now outside the window in both directions, and clique/pivot
cells remain below the historical baseline window.

Per S_INT.3, M_INT.5-M_INT.11 remain not run until M_INT.4 is amended or the
remaining W5.2 corpus mismatch is resolved.
