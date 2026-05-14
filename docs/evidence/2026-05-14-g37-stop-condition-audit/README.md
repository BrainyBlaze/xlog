# G37 Stop-Condition Audit

**Goal document:** `docs/plans/2026-05-13-supervisor-goal-037.md`
**Audit branch:** `docs/g37-stop-condition-audit`
**Audit base:** `feat/w38-stream-mux-aot-g37 @ 792cea72`
**Status:** G37 is not complete; G2 and G3 gates remain red.

## Objective Restated

G37 is complete only when W3.3, W3.5, W3.6, W3.7, W3.8, W3.9, and cross-cutting cleanup all satisfy the metrics in the goal document, with a closure proposal and explicit user approval before any board state change.

## Prompt-To-Artifact Checklist

| Requirement | Expected artifact or command | Current evidence | Status |
|---|---|---|---|
| G1 / W3.3 metrics M1.1-M1.7 | `feat/w33-hg-block-slice-prod` plus evidence README | `035b0713`, `docs/evidence/2026-05-13-w33-hg-block-slice-prod/README.md` | GREEN artifact present |
| G2 / W3.5 M2.1 speedup >= 1.5x | passing `cargo bench --bench wcoj_w35_smallrel` spike, then production branch | `50f06d25`, `9b468216`, `60b585e9`; best current G37-local result is red | RED |
| G2 / W3.5 M2.2 no regression above threshold | same W3.5 bench large cell | `9b468216` large `0.510x`; `60b585e9` large `0.526x` | RED |
| G2 / W3.5 M2.3 row equality | W3.5 spike cert in bench setup | PASS in `9b468216` and `60b585e9` | GREEN |
| G2 / W3.5 M2.4 shared-memory budget <= 32 KB | occupancy/resource evidence | `60b585e9` uses `4,352` static bytes; `c53dce32` confirms resource profile | GREEN |
| G3 / W3.6 M3.1 speedup >= 1.3x | passing `cargo bench --bench wcoj_w36_warp` spike, then production branch | `6c396757` small `0.595x` | RED |
| G3 / W3.6 M3.2 no regression above threshold | same W3.6 bench large cell | `6c396757` large `0.521x` | RED |
| G3 / W3.6 M3.3 row equality | W3.6 bench cert | PASS in `6c396757` | GREEN |
| G4 / W3.7 M4.1-M4.5 | helper-split production branch plus evidence | `feat/w37-helper-split-aot-g37 @ bfd80d67` | GREEN artifact present |
| G5 / W3.8 M5.1-M5.4 | stream-mux production branch plus evidence | `feat/w38-stream-mux-aot-g37 @ 792cea72` | GREEN artifact present |
| G6 / W3.9 production-scale suite | `wcoj_paper_class` bench and >= 3 fixtures | no branch/artifact yet | MISSING |
| G7 cleanup gates | cleanup/audit evidence, dead-code follow-up | no bundle integration branch yet | MISSING |
| W3.4 re-validation | superhub-50K >= 1.3x post-bundle | not run after G2/G3 because those gates are red | MISSING |
| W4.1 regression | 3 dispatch certs PASS on final bundle | not run on final bundle because final bundle does not exist | MISSING |
| Closure proposal | `docs/plans/2026-05-XX-w3-bundle-closure-proposal.md` | not written | MISSING |
| Board update | user-approved board edit after proposal | not authorized | NOT STARTED |

## G2 Evidence Summary

| Branch | Commit | Candidate | Small cell | Large cell | Row equality | Budget |
|---|---|---|---:|---:|---|---|
| `bench-spike/w35-shmem-narrow-g37` | `50f06d25` | dynamic tile + row map | `0.501x` | `0.554x` | PASS | PASS |
| `bench-spike/w35-shmem-narrow-paired-g37` | `9b468216` | same candidate, balanced pairing | `0.521x` | `0.510x` | PASS | PASS |
| `bench-spike/w35-static-xz-tile-g37` | `60b585e9` | 64-value static `xz` tile | `0.498x` | `0.526x` | PASS | PASS |
| `bench-spike/w35-static-xz-tile-control-g37` | `c53dce32` | resource diagnostic | n/a | n/a | n/a | PASS |

Resource diagnostic from `c53dce32`:

```text
wcoj_triangle_count_hg_cached_u32:  REG:32 STACK:192 SHARED:5120 LOCAL:0
wcoj_triangle_count_hg_xz_tile_u32: REG:32 STACK:192 SHARED:5388 LOCAL:0
```

Interpretation: the static-tile miss is not explained by a register-count increase, stack increase, local-memory spill, or large shared-memory growth.

## G3 Evidence Summary

| Branch | Commit | Candidate | Small cell | Large cell | Row equality |
|---|---|---|---:|---:|---|
| `bench-spike/w36-warp-coop-g37` | `8808cdf0` | warp row-handle broadcast | `0.597x` | `0.515x` | PASS |
| `bench-spike/w36-warp-coop-paired-g37` | `6c396757` | same candidate, balanced pairing | `0.595x` | `0.521x` | PASS |

## Stop-Condition Finding

The G37 per-sub-goal loop requires another design iteration while a spike is red, and the bundle-level stop condition says a sub-goal with at least three failed redesigns is stuck and must be escalated. G2 now has three G37-local red spike measurements plus one resource diagnostic. G3 remains red after balanced pairing.

The measured fixture shape explains the miss: the fixed W3.5/W3.6 triangle cells expose only about two useful probes per `xy` row on average. Line-6 shared-memory setup and line-7 warp coordination add work that cannot amortize on this fixture.

## Required Supervisor Decision

G37 cannot proceed to production for W3.5 or W3.6 on the current acceptance cells. A supervisor decision is required before additional implementation:

1. Amend the W3.5/W3.6 fixture or metric to target a higher fan-out paper-class cell.
2. Authorize a new W3.5/W3.6 design target with a different measurable work shape.
3. Close G37 as blocked with the current evidence and issue a revised goal.

No board edit, DONE marking, merge, push, or tag is included in this audit.
