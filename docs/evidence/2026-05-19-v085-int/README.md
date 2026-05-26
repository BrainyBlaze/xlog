# v0.8.5 Integration Evidence

Sub-goal: `G085_INT`

## Scope

- Ran the full integration command list from
  `docs/plans/2026-05-18-agent-v085-language-completeness-goal.md`.
- Fixed a compatibility regression found by the integration gate: the v0.8.5
  list normalizer rejected ordinary legacy `pair/2` relations because the
  reserved pair-helper check was name-only.
- Added a regression test proving `pair/2` relations compile while reserved
  helper arities remain rejected.

## Checks

```text
cargo fmt --check
# PASS

cargo check -p xlog-logic
# PASS

cargo check -p xlog-prob
# PASS

cargo check -p xlog-cli
# PASS

cargo test -p xlog-logic
# PASS
# lib: 236 passed
# test_v085_lists: 8 passed, including ordinary_pair_arity_two_relation_is_not_reserved_list_helper
# doc-tests: 0 passed, 5 ignored

cargo test -p xlog-prob
# PASS
# lib: 56 passed
# test_v085_aggregate_lifting: 4 passed
# test_v085_prob_aggregates: 4 passed
# no_cpu_d4_in_exact/no_dtoh_gpu_native/no_dtoh_* source audits passed

cargo test -p xlog-cli
# PASS
# explain_cli_tests: 2 passed
# interactive_cli_tests: 2 passed
# run_cli_tests: 1 passed

cargo test -p xlog-runtime
# PASS
# lib: 125 passed
# statistics_tests: 3 passed
# test_w21_part_b: 7 passed
# test_w25_default_flip: 3 passed
# test_w67b_dispatch_plan_source: 2 passed
# doc-tests: 2 passed, 2 ignored

cargo test -p xlog-integration
# PASS after the pair/2 compatibility fix
# executor_config_tests: 9 passed
# e2e_integration_tests: 18 passed
# real_world_tests: 13 passed
# WCOJ, DTS analog, widened-frontier, and cross-mode determinism suites passed

python3 scripts/validate_v085_examples.py --output /tmp/v085_examples_validation_summary.json
# PASS, example_count=10, interaction_count=10, all per-example statuses PASS

pytest -q python/tests/test_v080_examples_source.py python/tests/test_v085_examples_source.py
# 4 passed

python3 -m json.tool docs/evidence/2026-05-19-v085-examples/validation_summary.json
# PASS

python3 -m json.tool /tmp/v085_examples_validation_summary.json
# PASS

git diff --check
# PASS

rg -n "TODO|FIXME|TBD|PLACEHOLDER|placeholder|stub|unimplemented" \
  docs/evidence/2026-05-19-v085-examples \
  examples/v085-language/showcase \
  scripts/validate_v085_examples.py \
  crates/xlog-logic/src/list_normalize.rs \
  crates/xlog-logic/tests/test_v085_lists.rs
# No matches; rg exit 1 as expected for an empty result set.
```

## Acceptance Notes

- `M085_INT.1`: formatting passed.
- `M085_INT.2`: `xlog-logic` tests passed after the compatibility regression
  test was added.
- `M085_INT.3`: `xlog-prob` tests passed, including GPU/native source audits
  and v0.8.5 probabilistic aggregate tests.
- `M085_INT.4`: `xlog-cli` tests passed for explain, REPL, watch, run, and the
  no-host-output default prob CLI surface.
- `M085_INT.5`: `xlog-runtime` and `xlog-integration` tests passed, including
  WCOJ and strict deterministic D2H gates.
- `M085_INT.6`: v0.8.5 examples validator passed.
- `M085_INT.7`: source-audit tests for unapproved CPU/D2H routes passed.
- `M085_INT.8`: v0.8.0 and v0.8.5 example source guards passed.
- `M085_INT.9`: JSON validation, placeholder scan, and diff whitespace checks
  passed.
