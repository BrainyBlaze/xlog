# G38 G_INT M_INT.1 Blocker

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.1 W3.4 re-validation
**Branch:** `feat/w3-bundle-integration`
**First observed code HEAD:** `caf54929`
**Blocker still active after:** `cc305412`

Related follow-up artifacts:

- `docs/evidence/2026-05-14-g38-completion-audit.md`
- `docs/plans/2026-05-14-g38-mint1-response-proposal.md`

## Required gate

M_INT.1 requires W3.4 re-validation on integration HEAD:

```text
cargo bench --bench wcoj_w34_kernel_fusion
```

Target: ratio >= 1.51x.

## Observed result

The named bench target does not exist on integration HEAD:

```text
cargo bench -p xlog-integration --bench wcoj_w34_kernel_fusion --no-run
EXIT 101

error: no bench target named `wcoj_w34_kernel_fusion` in `xlog-integration` package
```

The same command was rerun after adding the completion audit and response
proposal commits; it still exits 101 with the same missing-target error.

The older W3.4 implementation commit is an ancestor of integration HEAD:

```text
git merge-base --is-ancestor 70d2cf5e HEAD
EXIT 0
```

However, the W3.4 implementation added `wcoj_fusion_bench`, not
`wcoj_w34_kernel_fusion`, and later W33 commits removed the W3.4 fused-count
bench and cert surface:

```text
70d2cf5e feat(w34): production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
  crates/xlog-integration/benches/wcoj_fusion_bench.rs
  crates/xlog-cuda-tests/tests/test_wcoj_w34_fusion.rs

738ab6f2 feat(w33 G1/S1.4 M1.1): retire W3.4 fused count kernel
0754a30d feat(w33 G1/S1.4 M1.1 M1.3): retire old u32 triangle count surface
  deleted crates/xlog-integration/benches/wcoj_fusion_bench.rs
  deleted crates/xlog-cuda-tests/tests/test_wcoj_w34_fusion.rs
```

Current HEAD has neither the plan-named target nor the older W3.4 bench:

```text
rg --files crates/xlog-integration/benches crates/xlog-cuda-tests/tests | rg 'w34|fusion'
EXIT 1
```

## Verdict

G_INT stops at M_INT.1.

This is not a performance regression measurement yet; it is a missing
re-validation artifact. The response proposal expands the choices as an
explicit decision artifact; in short, the supervisor must choose one of:

1. Restore or replace the W3.4 re-validation bench and rerun M_INT.1.
2. Amend M_INT.1 to a successor metric that validates the post-W33 replacement
   for the retired W3.4 fused-count surface.
3. Treat G_INT as STUCK under the current goal-038 contract.

Later G_INT metrics were not run because S_INT.3 requires stopping on the first
failure.
