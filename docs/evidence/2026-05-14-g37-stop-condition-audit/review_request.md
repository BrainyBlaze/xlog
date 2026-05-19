# G37 Review Request

**Branch:** `docs/g37-stop-condition-audit`
**Prepared from:** `docs/g37-stop-condition-audit @ cd185e9e`
**Goal:** `docs/plans/2026-05-13-supervisor-goal-037.md`
**Status:** supervisor decision required before further G2/G3 implementation.

## Finding

G37 is not complete. Section 14.2 stop condition is active because G2/W3.5 has three red redesign measurements and G3/W3.6 remains red after corrected paired measurement.

## Current Evidence

| Item | Evidence | Status |
|---|---|---|
| G1 W3.3 | `feat/w33-hg-block-slice-prod @ 035b0713` | GREEN artifact present |
| G2 W3.5 M2.1 | `50f06d25=0.501x`; `9b468216=0.521x`; `60b585e9=0.498x`; target `>=1.5x` | RED |
| G2 W3.5 M2.2 | `50f06d25=0.554x`; `9b468216=0.510x`; `60b585e9=0.526x`; target within `+5%` | RED |
| G2 W3.5 M2.3/M2.4 | row equality PASS; `c53dce32` resource diagnostic PASS | GREEN |
| G3 W3.6 M3.1 | `8808cdf0=0.597x`; `6c396757=0.595x`; target `>=1.3x` | RED |
| G3 W3.6 M3.2 | `8808cdf0=0.515x`; `6c396757=0.521x`; target within `+5%` | RED |
| G3 W3.6 M3.3 | row equality PASS | GREEN |
| G4 W3.7 | `feat/w37-helper-split-aot-g37 @ bfd80d67` | GREEN artifact present |
| G5 W3.8 | `feat/w38-stream-mux-aot-g37 @ 792cea72` | GREEN artifact present |
| G6/W3.9, G7, W3.4 revalidation, W4.1 final regression, closure proposal, board update | no final-bundle artifact yet | MISSING |

## Root Cause

The current W3.5/W3.6 acceptance cells do not expose enough line-6/line-7 work to amortize shared-memory setup or warp coordination:

| Fixture fact | Value |
|---|---:|
| total HG work | `99,609` |
| avg work per `xy` row | `1.992` |
| p50 work per `xy` row | `2` |
| p90 work per `xy` row | `4` |
| max work per `xy` row | `8` |

The resource diagnostic `c53dce32` rules out register, stack, local-memory, and shared-memory footprint as the cause.

## Requested Decision

Choose one response before any further G2/G3 implementation:

| Response | Decision |
|---|---|
| Response 1 recommended | approve Iteration 2 fixture amendment: `triangle-line6-fanout-512` for G2 and `triangle-line7-warp-prefix-512` for G3 |
| Response 2 | authorize a different W3.5/W3.6 design target on the original cells |
| Response 3 | revise the G37 bundle DAG before any closure proposal |

## Response 1 Readiness

If Response 1 is approved, use:

| Step | Branch | Base | Worktree |
|---|---|---|---|
| G2 iteration 2 spike | `bench-spike/w35-line6-fanout-g37` | `feat/w33-hg-block-slice-prod @ 035b0713` | `.worktrees/w35-line6-fanout-g37` |
| G3 iteration 2 spike | `bench-spike/w36-line7-fanout-g37` | passing G2 iteration 2 HEAD | `.worktrees/w36-line7-fanout-g37` |

The readiness appendix is `docs/evidence/2026-05-14-g37-stop-condition-audit/response1_readiness.md`.

## Boundary

No implementation branch was cut. No board edit, DONE marking, merge, push, or tag is requested in this review request.
