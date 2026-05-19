# G_PURGE2 Phase-2 Purge Evidence

**Goal:** Goal-039 G_PURGE2, Phase-2 cross-cutting cleanup.
**Branch:** `feat/w6-bundle-integration-g39`
**Worktree:** `.worktrees/g39-w6-bundle-integration`
**Date:** 2026-05-18
**Input checkpoint:** G_INT2 commit `4d737ac5`
**Scope:** Source-code files touched by Phase 2, plus Phase-2 commit-message
audits. Historical plan archives and closure-board governance text imported from
main were preserved as records and were not rewritten by this purge.

## Result

G_PURGE2 is green for M_PURGE2.1 through M_PURGE2.9.

The purge changed only comments in:

```text
crates/xlog-cuda/src/provider/groupby.rs
crates/xlog-cuda/src/provider/relational.rs
crates/xlog-cuda/tests/set_ops_tests.rs
```

No dependency manifests, CUDA kernels, runtime behavior, dispatch logic, or
neural-symbolic API surface changed.

## Scope

The broad predecessor diff is:

```text
git diff --name-only --diff-filter=ACMRT c1689d70..HEAD
155 paths
```

For implementation-marker and future-version scans, G_PURGE2 uses the 85
source/build files in that set:

```text
git diff --name-only --diff-filter=ACMRT c1689d70..HEAD \
  | rg '\.(rs|cu|h|hpp|cuh|toml|py|pyi|sh)$|Cargo\.lock$'
85 paths
```

This keeps the purge focused on Phase-2 implementation artifacts. The broad diff
also contains historical supervisor-plan archives and closure-board text that
quote the process rules themselves; those files are evidence records, not
implementation comments.

## Metric Matrix

| Metric | Evidence | Result |
|---|---|---|
| M_PURGE2.1 implementation-marker comments | Source-scope marker scan from the plan produced no output. | PASS |
| M_PURGE2.2 churn comments | Source-scope churn-phrase scan from the plan produced no output after rephrasing legacy comments in `groupby.rs`, `relational.rs`, and `set_ops_tests.rs`. | PASS |
| M_PURGE2.3 unused deps | `cargo +nightly udeps --workspace --all-targets` exited `0` and ended with `All deps seem to have been used.` | PASS |
| M_PURGE2.4 strict dead-code build | `RUSTFLAGS="-D dead_code -D unused_imports -D unused_variables" cargo build --workspace --all-targets --release` exited `0`; release build finished in `1m 07s`. | PASS |
| M_PURGE2.5 co-author trailers | `git log --format='%H %s%n%B%x00' c1689d70..HEAD \| rg -n 'Co-Authored-By'` produced no output. | PASS |
| M_PURGE2.6 future-version marker | Phase-2 source-scope scan and commit-message scan produced no output for the forbidden future-version token. Historical governance docs that quote the lock remain preserved. | PASS |
| M_PURGE2.7 paper-citation coverage | `crates/xlog-cuda/kernels/wcoj.cu` carries the HG kernel citation: lines 19-20 map the flattened root-key workspace to SRDatalog Algorithm 2 and name preserved/dropped lines; line 1783 cites paper section 5 Algorithm 1 Phase 1 for histogram maintenance. | PASS |
| M_PURGE2.8 inherited dead-code follow-up | `docs/evidence/2026-05-14-g38-dead-code-followup.md` exists and remains the Phase-1 non-touched dead-code follow-up input. This purge found no compiled dead code in Phase-2 source scope. | PASS |
| M_PURGE2.9 graceful-close citation | Phase 2 did not invoke a new W35/W36 graceful-close path. The inherited kernel-header citation pattern is still present in `crates/xlog-cuda/kernels/wcoj.cu` lines 19-20. | PASS |

## Hygiene

```text
git diff --check
cargo fmt --check
```

Both commands exited `0`.
