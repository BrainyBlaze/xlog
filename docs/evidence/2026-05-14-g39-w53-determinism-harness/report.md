# G_W53 W5.3 Cross-Mode Determinism Harness Evidence

Branch: `feat/w53-determinism-harness-g39`

Base: `feat/g39-pre-profiler-trace` at `d7a82eca3fbfee09976559cb4354c1f5e8804621`

## Scope

Adds `crates/xlog-integration/tests/test_cross_mode_determinism.rs`.

The harness runs under `XLOG_DETERMINISTIC=1` with fixed fixture seed `1395`
(`0x573`) and validates:

- forced WCOJ triangle execution,
- forced binary-join fallback execution,
- recursive-SCC execution over the same triangle + chain + transitive-closure
  fixture,
- dynamic rule injection by recompiling R1 with R2,
- Stage 5 -> Stage 4 arm-D rollback by simulating
  `StrictTrainResult.discovered_rule` and injecting it.

## Metrics

| Metric | Status | Raw Evidence |
|---|---:|---|
| M_W53.1 | PASS | `crates/xlog-integration/tests/test_cross_mode_determinism.rs` exists; targeted test binary has 2 tests |
| M_W53.2 | PASS | 3-way canonical row equality: WCOJ == binary fallback == recursive SCC for `tri`, `chain`, `path`; `wcoj_dispatches=1`, `binary_dispatches=0`, `recursive_dispatches=1` |
| M_W53.3 | PASS | `iterations=100`; fixed-seed WCOJ evaluate loop remained bit-exact; `tri_rows=5`, `chain_rows=4`, `path_rows=9` |
| M_W53.4 | PASS | dynamic R1 -> R1+R2 injection loop `iterations=100`; `dynamic_before_rows=8`, `dynamic_after_rows=9` |
| M_W53.5 | PASS | Stage 5 rollback simulator loop `iterations=100`; `stage5_before_rows=8`, `stage5_after_rows=9` |
| M_W53.6 | PASS | no nondeterminism observed; RCA doc not needed |

## Targeted Run

Command:

```bash
cargo test -p xlog-integration --test test_cross_mode_determinism -- --nocapture
```

Output summary:

```text
running 2 tests
G_W53_CROSS_MODE fixed_seed=1395 iterations=100 wcoj_dispatches=1 binary_dispatches=0 recursive_dispatches=1 tri_rows=5 chain_rows=4 path_rows=9
test cross_mode_wcoj_binary_recursive_outputs_are_bit_exact_100x ... ok
G_W53_INJECTION fixed_seed=1395 iterations=100 dynamic_before_rows=8 dynamic_after_rows=9 stage5_before_rows=8 stage5_after_rows=9
test dynamic_injection_and_stage5_rollback_are_bit_exact_100x ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## TDD Red Gate

Initial sentinel command:

```bash
cargo test -p xlog-integration --test test_cross_mode_determinism -- --nocapture
```

Expected red output:

```text
test g_w53_harness_red_sentinel ... FAILED
G_W53 cross-mode determinism harness is not implemented yet
test result: FAILED. 0 passed; 1 failed
```
