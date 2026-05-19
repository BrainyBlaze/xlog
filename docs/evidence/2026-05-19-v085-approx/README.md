# v0.8.5 Approximate Inference Evidence

Sub-goal: `G085_APPROX`

## Scope

- Added source pragmas for MC sample count, seed, confidence, sampling method,
  and nonmonotone iteration cap.
- Added `McEvalConfig::from_directives` so source directives flow into the
  existing MC engine instead of duplicating configuration logic in the CLI.
- Updated `xlog prob` so source `#pragma prob_engine = mc` selects MC by
  default, while CLI flags override source pragmas field-by-field.
- Added MC JSON output and extended pretty/csv/arrow MC batches with sample
  count, evidence count, seed, confidence, and sampling method columns.

## Certified Fixture

`examples/v085-language/approx/aggregate_mc.xlog` configures MC entirely through
source pragmas and runs a probabilistic count aggregate in approximate mode.

## Checks

```text
cargo test -p xlog-logic --test test_v085_approx_pragmas
# 2 passed

cargo test -p xlog-prob --features host-io --test test_v085_approx
# 4 passed

cargo test -p xlog-cli --features host-io --test prob_cli_tests
# 2 passed

cargo test -p xlog-logic --lib
# 236 passed

cargo test -p xlog-prob --lib
# 56 passed

cargo check --workspace
# PASS

cargo fmt --check
# PASS

git diff --check
# PASS
```

## Acceptance Notes

- `M085_APPROX.1`: `prob_samples`, `prob_seed`, `prob_confidence`,
  `prob_method`, and `prob_max_nonmonotone_iterations` parse into directives.
- `M085_APPROX.2`: CLI MC flags override matching source pragmas; unprovided
  flags inherit source pragmas or engine defaults.
- `M085_APPROX.3`: Fixed-seed MC replay is certified with identical estimates
  and confidence intervals.
- `M085_APPROX.4`: MC JSON and batch outputs include probability, stderr,
  CI low/high, total samples, evidence samples, seed, confidence, and method.
- `M085_APPROX.5`: The approximate aggregate example passes through the MC
  engine with confidence metadata.
