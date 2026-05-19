# G38 G_INT M_INT.5 Default-Flip Cert

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.5 W2.5 default-flip cert
**Branch:** `feat/w3-bundle-integration`
**Status:** PASS after restoring a narrow W2.5 cost-model selector.

## Change

The G38 integration branch already contained W2.5, but the exact M_INT.5 gate
was uncovered: `cargo test -p xlog-runtime test_w25_default_flip` previously
ran zero tests. Later G1/S1.7 work also removed the adaptive skew-classifier
surface, so the written `env skew opt-out preserved` clause had no current API.

This checkpoint restores only the W2.5 selector surface:

```text
crates/xlog-core/src/config.rs
crates/xlog-core/src/lib.rs
crates/xlog-runtime/src/executor/wcoj_cost_model.rs
crates/xlog-runtime/tests/test_w25_default_flip.rs
```

`RuntimeConfig::default().resolved_wcoj_cost_model()` resolves to
`CostModelKind::Cardinality`. `XLOG_WCOJ_COST_MODEL=skew` resolves to
`CostModelKind::SkewClassifier`.

Because G1/S1.7 removed the GPU skew classifier kernels/provider surface, the
runtime `SkewClassifier` branch is a conservative opt-out from cardinality
dispatch instead of a restoration of deleted classifier scoring.

## TDD Red

Command:

```text
cargo test -p xlog-runtime test_w25_default_flip
```

Result before implementation:

```text
EXIT 101
error[E0432]: unresolved import `xlog_core::CostModelKind`
error[E0599]: no method named `resolved_wcoj_cost_model` found for struct `RuntimeConfig`
error[E0599]: no method named `with_wcoj_cost_model` found for struct `RuntimeConfig`
```

## Gate Rerun

Command:

```text
cargo test -p xlog-runtime test_w25_default_flip
```

Result after implementation:

```text
EXIT 0
running 2 tests
test executor::wcoj_cost_model::tests::test_w25_default_flip_factory_honors_env_skew_opt_out ... ok
test executor::wcoj_cost_model::tests::test_w25_default_flip_factory_uses_cardinality_default ... ok
test result: ok. 2 passed; 0 failed; 118 filtered out

running 3 tests
test test_w25_default_flip_env_skew_opt_out_preserved ... ok
test test_w25_default_flip_cardinality_is_default ... ok
test test_w25_default_flip_config_override_beats_env ... ok
test result: ok. 3 passed; 0 failed
```

## Additional Verification

```text
cargo test -p xlog-core --lib
EXIT 0
test result: ok. 27 passed; 0 failed
```

```text
cargo test -p xlog-runtime
EXIT 0
test result: ok. 120 passed; 0 failed
test result: ok. 3 passed; 0 failed
test result: ok. 7 passed; 0 failed
test result: ok. 3 passed; 0 failed
Doc-tests xlog_runtime: 2 passed; 0 failed; 2 ignored
```

## Verdict

M_INT.5 is green under the written command. The result preserves the selector
and env opt-out semantics, with the caveat that `skew` is now a conservative
post-G1 opt-out from cardinality dispatch rather than the deleted adaptive GPU
classifier path.
