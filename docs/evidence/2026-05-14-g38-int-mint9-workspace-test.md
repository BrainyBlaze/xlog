# G38 G_INT M_INT.9 Workspace Release Tests

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.9 workspace release tests excluding `pyxlog` and `xlog-cuda-tests`
**Branch:** `feat/w3-bundle-integration`
**Status:** PASS.

## Command

```text
cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests
```

## Initial Blocker

The first M_INT.9 run stopped at:

```text
tests/test_wcoj_4cycle_adaptive_dispatch.rs
adaptive_dispatches_on_superhub_fixture
assertion left == right failed
left: 0
right: 1
```

The test still encoded the removed adaptive skew-classifier contract: a
super-hub fixture alone was expected to dispatch. Current post-G1 behavior is
cardinality-backed: adaptive dispatch only fires when runtime relation
cardinalities are populated enough to cross the WCOJ cost-model threshold.

## Remediation

Updated `crates/xlog-integration/tests/test_wcoj_4cycle_adaptive_dispatch.rs`
so the positive adaptive case seeds large relation cards for `e1` through
`e4`, while the no-stats case remains the negative fallback check. The force
gate test remains a separate bypass check.

## Targeted Retest

```text
cargo test -p xlog-integration --release --test test_wcoj_4cycle_adaptive_dispatch -- --nocapture
```

```text
EXIT 0
running 4 tests
test adaptive_dispatches_on_superhub_fixture_with_seeded_cards ... ok
test force_gate_dispatches_regardless_of_adaptive ... ok
test adaptive_default_off_does_not_dispatch_on_superhub ... ok
test adaptive_falls_back_on_uniform_fixture ... ok
test result: ok. 4 passed; 0 failed
```

## Full Retest

```text
EXIT 0
```

The full workspace release test command completed with no failing test targets.

## Fresh Post-Fix Guards

```text
cargo fmt --check --all
EXIT 0

RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog
EXIT 0
Finished `release` profile [optimized] target(s) in 0.07s
```
