# G38 Dead-Code Follow-Up

**Branch:** `feat/w3-bundle-integration`
**Base:** `main @ f62188b7`
**G_INT checkpoint before purge:** `d339b7c2`
**Date:** 2026-05-14

## Result

G_PURGE is green for M_PURGE.1 through M_PURGE.9.

The purge changed only dependency manifests, `Cargo.lock`, and one
documentation comment in `RuntimeConfig`. It did not delete branch heads,
worktrees, preserved spike evidence, or downstream neural-symbolic API surface.

## Bundle-Touched Scope

The final touched-file scope is the output of:

```text
git diff --name-only --diff-filter=ACMRT f62188b7
```

At this purge checkpoint the scope contains 87 paths, including this evidence
artifact.

## M_PURGE Evidence

| Metric | Evidence | Result |
|---|---|---|
| M_PURGE.1 | Banned implementation-marker comment scan over the touched-file scope produced no output. | PASS |
| M_PURGE.2 | Banned churn-comment scan over the touched-file scope produced no output after rephrasing one `RuntimeConfig` doc comment. | PASS |
| M_PURGE.3 | `cargo +nightly udeps --workspace --all-targets` exited 0 and ended with `All deps seem to have been used.` | PASS |
| M_PURGE.4 | `RUSTFLAGS="-D dead_code -D unused_imports -D unused_variables" cargo build --workspace --all-targets --release` exited 0. | PASS |
| M_PURGE.5 | Bundle commit log scan found no co-author trailers. | PASS |
| M_PURGE.6 | Bundle commit and touched-file scans found no future-release marker references. | PASS |
| M_PURGE.7 | `crates/xlog-cuda/kernels/wcoj.cu` carries the Algorithm 2 / Phase-1 A5 citation on the HG kernel header. | PASS |
| M_PURGE.8 | This follow-up artifact exists. Strict compiler dead-code checking found no compiled-target dead code; preserved-unmerged branch heads are recorded below and were not removed. | PASS |
| M_PURGE.9 | G_W35 and G_W36 graceful-close evidence points to the required kernel-header paper citation. | PASS |

## Dependency Cleanup

`cargo +nightly udeps --workspace --all-targets` initially reported unused
manifest entries in five packages. The purge made these changes:

| Package | Change |
|---|---|
| `xlog-cli` | Made `xlog-prob` optional and tied it to the `host-io` feature that actually uses it. |
| `xlog-cuda-tests` | Removed unused `xlog-logic`, `xlog-runtime`, and `bytemuck` manifest entries. |
| `xlog-induce` | Removed unused `xlog-runtime`. |
| `xlog-neural` | Removed unused `lru`; no neural predicate, network-registration, training, or Python binding surface changed. |
| `xlog-runtime` | Moved `xlog-logic` behind the `recursive-stats-trace` feature and removed unused `bytemuck`. |

Feature-surface checks after the cleanup:

```text
cargo test -p xlog-cli --features host-io --tests --no-run
cargo test -p xlog-runtime --features recursive-stats-trace --test test_w23_recursive_stats --no-run
```

Both exited 0.

## Preserved Branch Heads

M_PURGE.8 is documentation-only follow-up scope. It did not remove stale or
unmerged branch heads.

The sibling-worktree sweep still shows the preserved W35, W36, W37, and W38
G37 worktrees, and the active G38 integration worktree:

```text
bench-spike/w35-line6-fanout-g38
bench-spike/w35-shmem-narrow-g37
bench-spike/w35-shmem-narrow-paired-g37
bench-spike/w35-static-xz-tile-control-g37
bench-spike/w35-static-xz-tile-g37
bench-spike/w36-warp-coop-g37
bench-spike/w36-warp-coop-paired-g37
feat/w37-helper-split-aot-g37
bench-spike/w37-helper-split-hand-g37
feat/w38-stream-mux-aot-g37
bench-spike/w38-stream-mux-hand-g37
bench-spike/w38-stream-mux-recompute-g37
feat/w3-bundle-integration
```

The broader W3 spike family remains present as branch heads. No `git branch -d`,
`git branch -D`, `git worktree remove`, or prune operation was run.

## Non-Touched Dead-Code Follow-Up Entries

| Entry | Status |
|---|---|
| Compiled non-touched Rust dead code | None found by the strict all-targets release build. |
| Downstream M18 / M37-A neural-symbolic surface | Preserved. The purge did not alter `xlog-prob`, `pyxlog`, or the `xlog-neural` public API surface; only an unused third-party dependency was dropped from `xlog-neural`. |
| Preserved spike / G37 branch heads | Preserved as historical evidence and future forensic context. |
