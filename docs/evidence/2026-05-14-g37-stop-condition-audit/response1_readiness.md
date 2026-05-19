# Response 1 Readiness Appendix

**Goal document:** `docs/plans/2026-05-13-supervisor-goal-037.md`
**Decision path:** Response 1 from `docs/plans/2026-05-14-g37-iteration-2-request.md`
**Status:** readiness notes only; no implementation branch was cut.

## Current State Check

| Check | Result |
|---|---|
| Existing `line6` / `line7` / `fanout-512` branches | none found |
| Existing `line6` / `line7` / `fanout-512` source hits | none found |
| Stop-condition audit branch | `docs/g37-stop-condition-audit @ ea1ac02e` before this appendix |
| Required supervisor action | approve Response 1 before new G2/G3 spike work |

## Response 1 Branch Plan

| Step | Branch | Base | Worktree | Purpose |
|---|---|---|---|---|
| G2 iteration 2 spike | `bench-spike/w35-line6-fanout-g37` | `feat/w33-hg-block-slice-prod @ 035b0713` | `.worktrees/w35-line6-fanout-g37` | measure W3.5 line-6 shared-memory benefit on `triangle-line6-fanout-512` |
| G3 iteration 2 spike | `bench-spike/w36-line7-fanout-g37` | approved passing G2 iteration 2 HEAD | `.worktrees/w36-line7-fanout-g37` | measure W3.6 line-7 warp-prefix benefit on `triangle-line7-warp-prefix-512` |

Initial command after approval:

```bash
git worktree add .worktrees/w35-line6-fanout-g37 -b bench-spike/w35-line6-fanout-g37 feat/w33-hg-block-slice-prod
```

G3 should be cut only after G2 iteration 2 has a passing spike artifact, because M3.1 compares against the G2-only path.

## Prior Code Surfaces To Reuse

| Prior artifact | Commit | Files |
|---|---|---|
| W3.5 static tile spike scaffold | `60b585e9` | `crates/xlog-integration/benches/wcoj_w35_static_tile.rs`; `crates/xlog-cuda/kernels/wcoj.cu`; `crates/xlog-cuda/src/provider/wcoj_metadata.rs`; `crates/xlog-cuda/src/kernel_manifest_data.rs`; `crates/xlog-cuda/src/provider/mod.rs`; `crates/xlog-integration/Cargo.toml` |
| W3.6 paired warp benchmark scaffold | `6c396757` | `crates/xlog-integration/benches/wcoj_w36_warp.rs` |

The prior W3.5/W3.6 benches already provide:

| Capability | Source |
|---|---|
| provider setup with fixed launch stream | `wcoj_w35_static_tile.rs`, `wcoj_w36_warp.rs` |
| layout phase before timing | both prior benches |
| HG work-plan construction | both prior benches |
| row-set equality check via `download_triples` | both prior benches |
| paired timing with explicit first/second order balance | corrected G2/G3 paired benches |

## Fixture Contract

The new acceptance fixture should make line-6 and line-7 work measurable while preserving the shared-memory budget:

| Cell | Shape |
|---|---|
| `triangle-line6-fanout-512` | `xy` rows each map to a key with `512` matching `z` values in both child relations; p50 HG work per `xy` row is at least `128`; per-key bracket is at most `12,288` bytes; row equality PASS |
| `triangle-line7-warp-prefix-512` | same fixture family; G3 compares warp-prefix path against the passing G2-only path; row equality PASS |

Concrete generator shape:

| Parameter | Value |
|---|---:|
| root keys | `512` |
| `xy` rows per root key | `1` |
| child `z` fanout per root key | `512` |
| expected output rows | `262,144` |
| p50 work per `xy` row | `512` |
| child bracket bytes per key per `u32` column | `2,048` |

This keeps each per-key child bracket under the `12,288` byte W3.5 line-6 budget while raising useful work far above the current p50 `2` fixture.

## Acceptance Checks After Approval

| Check | Required result |
|---|---|
| G2 M2.1 | `triangle-line6-fanout-512` speedup `>= 1.5x` |
| G2 M2.2 guard | current small/large cells row equality PASS, budget PASS, ratios reported |
| G2 M2.3 | row-set equality PASS on all measured cells |
| G2 M2.4 | shared-memory occupancy `<= 32 KB` |
| G3 M3.1 | `triangle-line7-warp-prefix-512` speedup `>= 1.3x` vs G2-only path |
| G3 M3.2 guard | current small/large cells row equality PASS, ratios reported |
| G3 M3.3 | row-set equality PASS on all measured cells |

## Non-Action Boundary

This appendix does not authorize implementation. It records the exact first branch, fixture contract, and files to inspect after Response 1 approval.
