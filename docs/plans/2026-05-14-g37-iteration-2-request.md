# G37 Iteration 2 Request — W3.5/W3.6 Stop-Condition Resolution

**Base goal:** `docs/plans/2026-05-13-supervisor-goal-037.md`
**Prepared from:** `docs/g37-stop-condition-audit @ e069911a`; augmented by readiness appendix `8cbaced9` and status matrix refresh `57d58ddb`
**Status:** supervisor decision required before further G2/G3 implementation.

## Stop-Condition Trigger

G37 section 14.2 defines the bundle as stuck when a sub-goal spike fails at least three consecutive redesigns. G2/W3.5 now meets that condition:

| Attempt | Commit | Candidate | M2.1 small cell | M2.2 large cell | Row equality | M2.4 |
|---|---|---|---:|---:|---|---|
| S2.1 dynamic tile | `50f06d25` | contiguous `xz` tile plus row map | `0.501x` | `0.554x` | PASS | PASS |
| balanced rerun | `9b468216` | same algorithm, corrected pairing | `0.521x` | `0.510x` | PASS | PASS |
| static tile | `60b585e9` | 64-value static `xz` tile | `0.498x` | `0.526x` | PASS | PASS |

G3/W3.6 is also red after corrected measurement:

| Attempt | Commit | Candidate | M3.1 small cell | M3.2 large cell | Row equality |
|---|---|---|---:|---:|---|
| S3.1 warp row-handle broadcast | `8808cdf0` | original paired bench | `0.597x` | `0.515x` | PASS |
| balanced rerun | `6c396757` | corrected pairing | `0.595x` | `0.521x` | PASS |

## Root-Cause Finding

The fixed W3.5/W3.6 cells do not contain enough line-6/line-7 work to amortize shared-memory setup or warp coordination:

| Fixture fact | Value |
|---|---:|
| total HG work | `99,609` |
| avg work per `xy` row | `1.992` |
| p50 work per `xy` row | `2` |
| p90 work per `xy` row | `4` |
| max work per `xy` row | `8` |

The `c53dce32` resource diagnostic rules out a compiler-resource cliff:

```text
wcoj_triangle_count_hg_cached_u32:  REG:32 STACK:192 SHARED:5120 LOCAL:0
wcoj_triangle_count_hg_xz_tile_u32: REG:32 STACK:192 SHARED:5388 LOCAL:0
```

The measurable issue is the work shape, not correctness, memory budget, row equality, register pressure, stack growth, or local-memory spills.

## Proposed Iteration 2 Direction

Replace the G2/G3 acceptance cells with a high-fan-out paper-class line-6/line-7 cell while retaining the current small/large cells as guard cells.

Implementation readiness is pinned in `docs/evidence/2026-05-14-g37-stop-condition-audit/response1_readiness.md`: no existing `line6` / `line7` / `fanout-512` branches or source hits were found; the first approved branch is `bench-spike/w35-line6-fanout-g37` from `feat/w33-hg-block-slice-prod @ 035b0713`; G3 waits for a passing G2 iteration-2 spike because M3.1 compares against the G2-only path.

### New G2/G3 Acceptance Cells

| Cell | Purpose | Required shape |
|---|---|---|
| `triangle-line6-fanout-512` | Proves W3.5 shared-memory line-6 benefit | per-key bracket fits within `12,288` bytes; p50 work per `xy` row at least `128`; row equality PASS |
| `triangle-line7-warp-prefix-512` | Proves W3.6 cooperative prefix benefit | same fixture family; p50 work per `xy` row at least `128`; row equality PASS |
| current `triangle-small-inner-4K` | guard cell | no correctness regression; budget PASS; ratio reported but not the acceptance ratio |
| current `triangle-large-yz-200K` | guard cell | no correctness regression; budget PASS; ratio reported but not the acceptance ratio |

### Revised Metrics

| Metric | Iteration 1 | Iteration 2 request |
|---|---|---|
| M2.1 | `>= 1.5x` on `triangle-small-inner-4K` | `>= 1.5x` on `triangle-line6-fanout-512` |
| M2.2 | within `+5%` on `triangle-large-yz-200K` | current small/large cells remain guard cells with row equality PASS and budget PASS; ratios reported |
| M3.1 | `>= 1.3x` on post-G2 small fixture | `>= 1.3x` on `triangle-line7-warp-prefix-512` vs G2-only path |
| M3.2 | within `+5%` above threshold | current small/large cells remain guard cells with row equality PASS; ratios reported |

## Response Options

**Response 1 — Accept Iteration 2 fixture amendment (recommended).**
Authorize new spike branches `bench-spike/w35-line6-fanout-g37` and `bench-spike/w36-line7-fanout-g37` from `feat/w33-hg-block-slice-prod`, using the high-fan-out cells above. Current red cells become guard evidence, not acceptance cells.

**Response 2 — Authorize a different design target.**
Keep the original cells, but permit a different W3.5/W3.6 mechanism that changes the work shape enough to make line-6/line-7 measurable on p50 `2` work. This is higher risk because three line-6 shared-memory variants already missed.

**Response 3 — Revise the bundle DAG.**
Keep G2/G3 red as documented evidence, continue only the already-green G1/G4/G5 path into W3.9, and explicitly revise the G37 completion definition before any closure proposal.

## Requested Supervisor Decision

Choose one response before any further G2/G3 implementation. No board edit, DONE marking, merge, push, or tag is requested here.
