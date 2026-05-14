# G38 M_INT.4 Response Proposal

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Branch:** `feat/w3-bundle-integration`
**Prepared after:** `docs/evidence/2026-05-14-g38-int-mint4-rca.md` and
`docs/evidence/2026-05-14-g38-int-mint4-e2-prefix-attempt.md`
**Status:** supervisor decision required; M_INT.4 remains red under the current
contract.

## Problem

M_INT.4 requires every W5.2 corpus ratio to be within `+-10%` of the historical
W5.2 closure baseline. Two facts now need to be separated:

1. The G38-only 4-cycle slowdown was real and has a production mitigation.
2. The historical W5.2 ratio window is not reproduced by the old W5.2 branch on
   the current machine, and the post-mitigation 4-cycle ratios are now faster
   than the historical `+10%` bound while `5clique` / `pivot5` remain below the
   historical `-10%` bound.

## Evidence

| Evidence | Result |
|---|---|
| `docs/evidence/2026-05-14-g38-int-mint4-rca.md` | Old W5.2 branch same-machine rerun misses historical baseline; G38 had an additional large 4-cycle regression. |
| `docs/evidence/2026-05-14-g38-int-mint4-e2-prefix-attempt.md` | E2-prefix production mitigation removes the G38-only large 4-cycle slowdown; M_INT.4 still fails the literal historical-ratio window. |
| `cargo test -p xlog-cuda --test test_w33_hg_source_audit --release -- --nocapture` | 7/7 PASS after adding the 4-cycle E2-prefix source guard. |
| `cargo test -p xlog-cuda --test test_wcoj_4cycle_u32 --release -- --nocapture` | 5/5 PASS after mitigation. |
| `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` | EXIT 0 with parity lines; 12/12 cells outside the historical `+-10%` ratio window. |

## Recommended Response

**Response 1 — Amend M_INT.4 to successor/no-regression criteria.**

Replace the literal historical ratio-window cell with:

1. Same-machine predecessor comparison against
   `/home/dev/projects/xlog/.worktrees/w52-skewed-multiway-bench @ 8941c487`
   for the W5.2 corpus.
2. For each cell, require row equality and no slower WCOJ GPU time than the
   same-machine W5.2 branch by more than `+10%`.
3. For ratio reporting, keep historical W5.2 medians as context only, not as a
   hard acceptance window, because they are not reproducible on the current
   machine.
4. Keep the new 4-cycle E2-prefix source guard as a regression test so the
   O(N^3)-shape HG work-item mapping cannot return.

Under this amendment, the post-mitigation 4-cycle cells have strong evidence of
successor improvement. `5clique` / `pivot5` should be judged against the
same-machine predecessor because their current values track the old branch rerun
more closely than the historical baseline.

## Alternatives

**Response 2 — Continue implementation under the literal historical window.**

Investigate `5clique` / `pivot5` and any remaining ratio-window mismatch until
all cells land within `+-10%` of the old W5.2 historical medians. This may force
performance shaping toward stale measurements rather than preventing
regression, because faster 4-cycle cells also fail the current literal window.

**Response 3 — Mark G38 STUCK at M_INT.4.**

Keep the current M_INT.4 contract unchanged and treat G38 as blocked even though
the G38-only 4-cycle regression has been mitigated.

## Not Included

This proposal does not mark G_INT green, does not start G_PURGE, does not edit
the closure board, and does not prune preserved-unmerged branch heads.
