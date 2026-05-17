# G38 G_INT M_INT.8 Workspace Build

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.8 workspace build with `-D warnings`
**Branch:** `feat/w3-bundle-integration`
**Status:** PASS.

## Command

```text
RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog
```

## Result

```text
EXIT 0
Finished `release` profile [optimized] target(s) in 31.81s
```

The build included workspace crates through `xlog-cli`, `xlog-integration`,
and `xlog-cuda-tests`, with `pyxlog` excluded per the gate.

## Refresh After M_INT.9 Test Update

```text
RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog
EXIT 0
Finished `release` profile [optimized] target(s) in 0.07s
```

## Refresh After M_INT.11 Instrumentation

```text
RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog
EXIT 0
Finished `release` profile [optimized] target(s) in 0.06s
```
