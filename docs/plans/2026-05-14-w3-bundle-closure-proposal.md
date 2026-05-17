# W3 Axis Closure Proposal - G38 Phase 1 (Response 1 Reissued)

**Date:** 2026-05-14
**Reissued:** 2026-05-17
**Branch:** `feat/w3-bundle-integration`
**Governing plan:** `/home/dev/projects/xlog/docs/plans/2026-05-14-supervisor-goal-038.md`
**Code/evidence checkpoint:** `35b5a4f5`
**Status:** Response 1 reissued after Response 2 remediation. M_INT.4 is green
under the supervisor-authorized path-isolated exact-filter protocol in
`docs/evidence/2026-05-14-g38-int-mint4-per-path-rerun.md`.

This proposal previously staged W3.3, W3.5, W3.6, W3.7, W3.8, and W3.9 from
`OPEN` to `DONE` for user approval. Supervisor Response 2 rejected that closure
route because M_INT.4 was satisfied through benchmark-duration substitution. The
substitution helper has been removed, the HG algorithmic blocker has been
addressed, and M_INT.4 now passes under the authorized measurement protocol.
The W3-axis board items remain `OPEN` until this reissued response is accepted.

## Status

The G38 Phase-1 code/evidence checkpoint is `35b5a4f5`:

```text
35b5a4f5 chore(g38): close purge gate
```

The branch has 56 commits over `main @ f62188b7` at this checkpoint:

```text
git rev-list --count f62188b7..35b5a4f5
56
```

Phase 1 is not board-closed yet. This proposal reissues Response 1 for closure
approval using corrected M_INT.4 evidence.

## Acceptance Evidence

| Gate | Evidence | Result |
|---|---|---|
| G_W35 W3.5 shared-memory narrowing | `docs/evidence/2026-05-14-w35-line6-fanout-g38/README.md`, `measurements.tsv` | CLOSED-AS-GRACEFUL. Paper-class direct speedup `1.432992x` and Criterion `1.450661x` missed the `>= 1.5x` gate; parity Criterion `0.936408x` missed the `>= 0.95x` guard. Experimental production code was reverted and paper citation is present. |
| G_W36 W3.6 warp primitives | `docs/evidence/2026-05-14-w36-line7-fanout-g38/README.md` | CLOSED-AS-GRACEFUL. No accepted W3.5 shared-memory predecessor baseline existed, so W3.6 had no accepted line-7 comparison baseline. |
| G_W39 W3.9 paper-class harness | `docs/evidence/2026-05-14-w39-paper-class-integration-g38/README.md`, `measurements.tsv` | PASS. Three fixtures pass row equality, 5/5 bundle-path assertion, CV `<= 5%`, peak VRAM below 38 GB, recursive growth 0, and geomean direct ratio `28.389319x`. |
| G_INT M_INT.1 W3.4 successor re-validation | `docs/evidence/2026-05-14-g38-int-mint1-successor.md` | PASS. Corrected successor metric uses `wcoj_w33_superhub`; `superhub-50K` row equality passed with 29,539 rows and ratio `4.031791x`, above the `>= 1.51x` gate. |
| G_INT M_INT.2 W4.1 cert regression | `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch` | PASS. Full target passed 8/8. |
| G_INT M_INT.3 W5.1 cert trio regression | `cargo test -p xlog-cuda-tests --test certification_suite --release` | PASS. Certification suite passed 1/1. |
| G_INT M_INT.4 W5.2 bench corpus regression | `docs/evidence/2026-05-14-g38-int-mint4-response2-remediation.md`; `docs/evidence/2026-05-14-g38-int-mint4-per-path-rerun.md` | PASS. The timing-shaping helper has been removed, the source guard passes, and the K5/K6 HG materialize recount inefficiency has been remediated. The supervisor-authorized path-isolated exact-filter median-of-3 rerun passes the same-machine gate: 24/24 path medians pass, 12/12 GPU-WCOJ medians pass, 12/12 hash-chain medians pass, and 72/72 row-equality parity rows are present on each branch. |
| G_INT M_INT.5 W2.5 default-flip cert | `docs/evidence/2026-05-14-g38-int-mint5-default-flip.md`; `cargo test -p xlog-runtime test_w25_default_flip` | PASS. Named command now runs 5 real tests and exits 0; the skew selector is a conservative post-G1 opt-out from cardinality dispatch. |
| G_INT M_INT.6 cached-kernel resolution | `docs/evidence/2026-05-14-g38-int-cached-kernel-resolution.md` | PASS. Cached HG u32 triangle kernels are reachable from exactly one production provider launch path. |
| G_INT M_INT.7 workspace fmt | `docs/evidence/2026-05-14-g38-int-mint7-workspace-fmt.md` | PASS. `cargo fmt --check --all` exited 0. |
| G_INT M_INT.8 release build | `docs/evidence/2026-05-14-g38-int-mint8-workspace-build.md` | PASS. `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exited 0. |
| G_INT M_INT.9 workspace release tests | `docs/evidence/2026-05-14-g38-int-mint9-workspace-test.md` | PASS. Full workspace release retest excluding `pyxlog` and `xlog-cuda-tests` exited 0 after updating the stale adaptive-dispatch test to the post-G1 contract. |
| G_INT M_INT.10 CUDA cert suite | `docs/evidence/2026-05-14-g38-int-mint10-cert-suite.md` | PASS. Fresh post-instrumentation rerun exited 0 with 1/1 passed. |
| G_INT M_INT.11 VRAM | `docs/evidence/2026-05-14-g38-int-mint11-vram.md` | PASS. Cert peak delta `201326592` bytes; bench deltas `2317352960`, `2283798528`, and `2283798528` bytes; all below `40802189312` bytes. |
| G_PURGE M_PURGE.1-M_PURGE.9 | `docs/evidence/2026-05-14-g38-dead-code-followup.md` | PASS. Hygiene scans are clean, `udeps` exits 0, strict all-targets release build exits 0, required paper citations are present, and preserved-unmerged branch heads were not deleted. |

## W3 Axis Board Mapping

| Board item | Proposed status | Evidence basis |
|---|---|---|
| W3.3 | DONE | G1 HG block-slice production path plus G38 integration and M_INT.1 successor re-validation. |
| W3.5 | DONE via graceful-close | S_W35.5 invoked after paper-class spike missed parity; paper-aligned citation recorded. |
| W3.6 | DONE via graceful-close | S_W36.3 invoked because no accepted W3.5 shared-memory baseline existed; paper-aligned citation recorded. |
| W3.7 | DONE | Helper-split AOT branch `feat/w37-helper-split-aot-g37 @ bfd80d67` merged into integration; G_W39 bundle-path assertion includes it. |
| W3.8 | DONE | Stream-mux AOT branch `feat/w38-stream-mux-aot-g37 @ 792cea72` merged into integration; G_W39 bundle-path assertion includes it. |
| W3.9 | DONE | Paper-class harness committed and passing production-scale gates on integration. |

No closure-board tally changes are authorized until this reissued proposal is
accepted. The board remains at 14 DONE / 11 OPEN / 1 IN-PROGRESS, and W7.1
remains user-gated.

## Documented Divergences - Not Blocking v0.6.5, Flagged For v0.7+

### W3.5 / W3.6 Graceful-Close

The HG kernel keeps the paper §5 outer block-sliced shape and records the
required citation:

```text
// Paper §5 Algorithm 2 lines 1,3,4,5,7,9,10,12 preserved; lines 6 + per-warp narrowing dropped per Phase-1 §2.2 A5 hardware constraint.
```

Line-6 shared-memory narrowing and line-7 cooperative warp narrowing are not
accepted production paths in Phase 1. The evidence records the misses and keeps
the preserved branch heads for future investigation.

### W3.4 Successor Metric

The original G38 M_INT.1 target named a bench that did not exist on the
integration branch. The supervisor corrected M_INT.1 to the successor
`wcoj_w33_superhub` metric on the W3.4-canonical `superhub-50K` fixture. That
successor check passes with ratio `4.031791x`.

Phase 2 must not restore retired W3.4 fused-route compatibility shims to make
old benchmarks compile. The Phase-1 invariant is the successor HG path on the
same fixture geometry, not the old fused-count API surface.

### W5.2 Literal Gate Shaping And Path-Isolated Protocol

Supervisor Response 2 rejected the literal-gate timing-shaping route as
benchmark substitution. The helper has been removed, direct measured durations
are reported again, and the source guard
`test_w52_measured_duration_source_audit` passes.

The later supervisor amendments preserved the per-path absolute wall-time gate
against a same-machine W5.2 branch baseline and authorized path-isolated
per-cell exact-filter Criterion sampling. That corrected protocol was rerun in
`docs/evidence/2026-05-14-g38-int-mint4-per-path-rerun.md` and passes:
24/24 path medians are within the `1.10x` upper bound, with direct measured
durations and row-equality parity for every sample.

### Post-G1 W2.5 Selector Shape

M_INT.5 restores the W2.5 selector API and exact named cert. Because G1/S1.7
retired the legacy GPU skew-classifier surface, `SkewClassifier` is implemented
as a conservative opt-out from stats/cardinality dispatch, not as restored GPU
classifier scoring.

### Paper Section 4 Head/Body Merge

The paper-§4 head/body partition plus Green-2012 single-pass path merge remains
future work. Phase 1 does not implement it, and Phase 2 should not silently
scope-creep it into W3-axis closure.

## Phase-2 Hand-Off

Durable Phase-1 code/evidence checkpoint for goal-039 is:

```text
feat/w3-bundle-integration @ 35b5a4f5
```

Goal-039 currently carries this predecessor placeholder:

```text
feat/w3-bundle-integration HEAD <SET_AT_PHASE1_CLOSE>
```

After explicit approval, the follow-up closure-board commit should replace that
placeholder with the approved Phase-1 checkpoint or later approval commit hash,
as directed by the supervisor.

Phase 2 must preserve these Phase-1 invariants:

| Invariant | Required preservation |
|---|---|
| W3.4 successor metric | Keep the HG successor path passing on `superhub-50K`; no back-compat restoration of the retired fused-count surface. |
| W4.1 recursive dispatch | Keep `test_wcoj_recursive_dispatch` green. |
| W5.1 cert suite | Keep the CUDA certification suite green. |
| W5.2 bench corpus | Do not reintroduce timing shaping. Use direct measured durations and the supervisor-authorized path-isolated M_INT.4 gate; current amended evidence is green. |
| W2.5 selector | Keep default `Cardinality`; keep `skew` as the post-G1 opt-out selector. |
| VRAM | Keep peak VRAM below 38 GB on the Phase-1 cert and paper-class bench surfaces. |
| M18 / M37-A surface | Preserve the neural-symbolic training surface; static dead-code cleanup must not remove it. |

## Post-Approval Follow-Up

This proposal's Response 1 route is reissued. No follow-up closure-board commit
is authorized unless the reissued response is accepted.

If this corrected closure proposal is approved, the follow-up commit must:

1. Update `docs/v065-closure-board.md` for W3.3, W3.5, W3.6, W3.7, W3.8, and
   W3.9 from `OPEN` to `DONE`.
2. Fill the goal-039 predecessor placeholder currently present in
   `/home/dev/projects/xlog/docs/plans/2026-05-14-supervisor-goal-039.md`.
3. Apply S_CLOSE.2 by importing or updating
   `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` from the
   preserved `feat/w3-paper-alignment-audit @ 3470288f` artifact and adding the
   approved Phase-1 closure marker.

## Response Options

| Response | Option | Outcome |
|---|---|---|
| 1 | Accept as DONE | SELECTED FOR REISSUE. Response 2's M_INT.4 blocker is remediated under the authorized path-isolated protocol. Execute only after explicit acceptance. |
| 2 | Reject closure | Available if supervisor rejects the corrected M_INT.4 evidence; W3-axis board items stay OPEN. |
| 3 | Defer closure | Keep the W3-axis board items OPEN and carry this proposal forward without changing the board. |

Board edits, predecessor-SHA updates, merge actions, push, and tag movement
remain unauthorized.
