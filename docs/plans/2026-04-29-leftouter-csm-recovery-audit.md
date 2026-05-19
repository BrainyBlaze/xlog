# Sub-slice 3 LeftOuter CSM — recovery audit

This audits `.recovery/sub-slice-3-edits.md` (the edits from
the lost commit `b90ae77f`) against the current
post-v0.6.0 access-aware contract documented in
`docs/architecture/recorded-launch-migration.md`. The lost
commit predates PR #72; several of its patterns no longer
match the current API and must be reworked before the
operator can land.

**Scope of this slice**: provider method + `unmatched_mask`
kernel + tests only. **No env dispatch wiring, no indexed
LeftOuter, no runtime wiring** — those are deferred to
follow-up sub-slices per the user direction.

## Files to apply (cleanly)

These three are pure additions, fully compatible with the
current API:

  * `crates/xlog-cuda/kernels/join.cu` — adds
    `hash_join_csm_unmatched_mask` kernel.
  * `crates/xlog-cuda/src/kernel_manifest_data.rs` — adds
    `"hash_join_csm_unmatched_mask"` to two slice arrays.
  * `crates/xlog-cuda/src/provider/mod.rs` — adds
    `pub const HASH_JOIN_CSM_UNMATCHED_MASK: &str = ...;`
    in two locations (matches the recovery doc).

The only judgement call here: do the manifest's two slice
arrays still exist post-v0.6.0? The recovery doc shows them
present at the lost-commit time. We verify post-v0.6.0
before applying.

## File that needs rework: provider method

`crates/xlog-cuda/src/provider/relational.rs` —
`hash_join_left_outer_v2_count_scan_materialize_recorded`.

The recovery's structure is correct (Phase A count-scan-total,
Phase B materialize matched indices, Phase C unmatched mask,
Phase C tail compact, Phase D gather matched, Phase E
per-column concat) but it uses three patterns that the
current contract removed or changed:

### Diff 1: `write_post_preflight_fresh` — removed API

The recovery calls `write_post_preflight_fresh` 7 times:

  * Phase A (line 448-451): `per_probe_count`,
    `per_probe_offsets`, `d_logical_count`, `d_overflow`.
  * Phase B (line 535-536): `d_output_left`, `d_output_right`.
  * Phase C (line 576): `d_unmatched_mask`.

The current contract: register every fresh runtime-backed
output with `rec.write(slice)` BEFORE
`rec.preflight(runtime)`. The recorder snapshots `BlockId` at
record time and drops the slice borrow immediately, so a
later `&mut slice` in a kernel param list is unaffected.

**Action**: in each phase, move the fresh-output `write`
calls above the corresponding `preflight` call.

### Diff 2: pre-preflight `cuMemsetD8Async` /
       `cuMemcpyDtoDAsync_v2` — needs reorder

Phase A (lines 319-344): zero-init `d_overflow` +
`d_logical_count` via `cuMemsetD8Async` BEFORE
`rec_count.preflight(...)`. With the current contract this is
unfenced — the memset launches on `cu_stream` before
preflight queues `cuStreamWaitEvent` against the alloc-ready
events. Pattern is exactly the bug class PR #72 fixed in
`hash_join_inner_v2_count_scan_materialize_recorded`.

Phase A (lines 398-411): `cuMemcpyDtoDAsync_v2` from
`per_probe_count` to `per_probe_offsets` BEFORE
`rec_count.preflight(...)`. Same issue.

**Action**: register all these buffers as writes before
preflight, MOVE the memsets/memcpy AFTER preflight — same
pattern as the now-correct
`hash_join_inner_v2_count_scan_materialize_recorded` (which
also does `prepare_first_use(d_overflow, ..., Access::Write)`
+ `prepare_first_use(d_logical_count, ..., Access::Write)`
right after the allocs). Either pattern works; the
recorder-managed approach (register-as-write then memset
post-preflight) is cleaner because it keeps all the
synchronization in one recorder commit.

### Diff 3: Phase E `runtime.record_block_use` — back-compat shim

Phase E records each output column via
`runtime.record_block_use(b, launch_stream)` (lines 738,
808). This is the v0.5 API; the current shim maps it to
`finish_block_use(Access::Read)`, which is functionally
correct for dealloc safety but semantically wrong (the
column was WRITTEN by the dtod-copy, not read).

**Action**: replace each `runtime.record_block_use(b, ls)`
with
`runtime.finish_block_use(BlockId::from_block(b), ls, Access::Write)`
for accurate access tagging. Matches the equivalent migration
on the other LeftOuter / indexed-LeftOuter sites in
`provider/relational.rs`.

### Diff 4: Phase E `out_col` allocs need `prepare_first_use`

The recovery's Phase E (lines 670-820) allocates `out_col`s
INSIDE a `rec_d` recorder window — `rec_d.preflight(...)` runs
before the alloc loop, then each `out_col` is allocated and
written via `cuMemsetD8Async` + `cuMemcpyDtoDAsync_v2` on
`cu_stream`. These `out_col`s are NEVER registered with
`rec_d` — they're written via raw CUDA calls and finished via
`runtime.record_block_use` at the end of each iteration.

The current contract: each `out_col` needs
`prepare_first_use(&out_col, launch_stream, Access::Write)`
immediately after `self.memory.alloc(...)`, BEFORE its first
memset/memcpy. Otherwise the alloc-ready event on the alloc
stream is not fenced into `launch_stream` and the memset can
run against pool-recycled bytes.

**Action**: add `prepare_first_use(...)` after each `out_col`
alloc. Pattern matches the post-PR-#72 fixes in
`hash_join_left_outer_v2_recorded` and `indexed left_outer`.

## Final action plan

1. Apply files 1-3 (kernel + manifest + module name) verbatim
   from recovery, after verifying the target slice arrays
   still exist in the current source.
2. Implement
   `hash_join_left_outer_v2_count_scan_materialize_recorded`
   from scratch using the recovery's structure as a reference
   but applying the four patches above. Take the
   already-correct
   `hash_join_inner_v2_count_scan_materialize_recorded` as the
   structural template (Phase A / B layout) and the
   already-correct
   `hash_join_left_outer_v2_recorded` as the Phase E concat
   template — both are post-PR-#72.
3. Add the 5 tests from recovery file #7 (lines 880+ in the
   recovery doc), but rewrite the harness boilerplate to
   match the current `test_provider_launch_recorder.rs`
   imports. Tests targeted:
     * Result-set correctness (matched + unmatched, multiset
       compare).
     * Drop+reuse partial-match.
     * Drop+reuse all-unmatched.
     * Empty-right legacy fallback.
     * Legacy-manager rejection.

## Out of scope (defer to follow-up sub-slices)

  * `hash_join_v2_recorded(LeftOuter, ...)` dispatch wiring —
    the operator is exposed only as the direct method in this
    slice; calling `hash_join_v2_recorded(LeftOuter)` still
    falls through to `hash_join_left_outer_v2_recorded` (the
    non-CSM LeftOuter variant) which is already in tree.
  * Indexed LeftOuter CSM
    (`hash_join_left_outer_v2_with_index_count_scan_materialize_recorded`)
    — separate slice.
  * Env-gated dispatch — separate slice.
