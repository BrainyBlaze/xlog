# v0.8.5 Aggregate Lifting Evidence

Sub-goal: `G085_AGG_LIFT`

## Scope

- Added exact small-domain lifting for finite probabilistic `count` aggregates.
- Count-only aggregate heads now build exact cardinality dynamic-programming PIR
  formulas instead of enumerating all row-presence masks.
- Production exact inference for small fired count-lift queries uploads the
  finite count-lift row probabilities and evaluates the cardinality DP in the
  `weights_count_lift_exact` CUDA kernel before falling back to the generic D4
  verifier path.
- Added provenance/explain metadata reporting whether lifting fired or fell back
  to exact finite outcome enumeration.
- Kept `sum`, `min`, `max`, and `logsumexp` on the existing exact finite
  enumeration path with explicit per-operator fallback reports.

## Certified Fixture

`examples/v085-language/aggregate_lifting/count_lift.xlog` uses 17 uncertain
contributing rows. Naive exact enumeration would require `2^17 = 131072`
outcomes for the group. The count lift reports 171 dynamic-programming states,
well over the required 1.5x structural cost reduction while preserving exact
finite-world semantics.

The production exact CLI path for the certified fixture routes through the
GPU-native count-lift path and returns the binomial oracle probability for
`out_degree(1, 8)`: `0.1854705810546875`.

## Checks

```text
cargo test -p xlog-prob --features host-io --test test_v085_aggregate_lifting
# 5 passed

cargo test -p xlog-cuda kernel_modules
# 2 passed

cargo test -p xlog-prob --test test_v085_aggregate_lifting
# 4 passed

CUDA_LAUNCH_BLOCKING=1 cargo run -q -p xlog-cli --features host-io -- \
  prob examples/v085-language/aggregate_lifting/count_lift.xlog --output json
# engine=exact_ddnnf; out_degree(1, 8)=0.1854705810546875

cargo test -p xlog-prob --test test_v085_prob_aggregates
# 4 passed

cargo test -p xlog-prob --features host-io --test test_v085_prob_aggregates
# 6 passed

cargo test -p xlog-prob --lib
# 56 passed

cargo test -p xlog-cli --test explain_cli_tests
# 2 passed

cargo check --workspace
# PASS

cargo check -p xlog-prob --features host-io
# PASS

cargo fmt --check
# PASS

git diff --check
# PASS
```

## Acceptance Notes

- `M085_AGG_LIFT.1`: `xlog explain --format json` emits
  `aggregate_lifting` entries with finite domain source, operator, status, cap,
  group key, and cost fields.
- `M085_AGG_LIFT.2`: `count` fires the lifted path. `sum`, `min`, `max`, and
  `logsumexp` report `fallback_exact_enumeration` with per-operator rationale.
- `M085_AGG_LIFT.3`: The 17-row count fixture matches a finite binomial oracle
  in both the aggregate-lift source test and the production exact CLI path, and
  the source test asserts the accepted exact fixture routes through the
  GPU-native count-lift evaluator. Existing numeric aggregate parity remains
  covered by exact enumeration tests.
- `M085_AGG_LIFT.4`: The certified count fixture reports `131072` naive
  outcomes versus `171` DP states.
- `M085_AGG_LIFT.5`: Count lift domains above 64 uncertain rows fail with a
  typed `v0.8.5 agg_lift error`.
