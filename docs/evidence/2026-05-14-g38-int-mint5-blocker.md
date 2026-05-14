# G38 G_INT M_INT.5 Blocker

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.5 W2.5 default-flip cert
**Branch:** `feat/w3-bundle-integration`
**Status:** SUPERSEDED by `docs/evidence/2026-05-14-g38-int-mint5-default-flip.md`.

## Gate Text

The governing goal-038 document names:

```text
M_INT.5 W2.5 default-flip cert
cargo test -p xlog-runtime test_w25_default_flip
PASS (Cardinality is default; env skew opt-out preserved)
```

## Fresh Command

Command:

```text
cargo test -p xlog-runtime test_w25_default_flip
```

Result:

```text
EXIT 0
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 118 filtered out
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out
```

This is a zero-test pass and does not certify M_INT.5.

## Coverage Check

The named test does not exist in the current integration branch:

```text
rg -n "test_w25_default_flip" crates/xlog-runtime crates/xlog-integration
EXIT 1
```

The W2.5 branch is already an ancestor of the integration branch, but later G1
work removed the legacy skew-classifier surface:

```text
git merge-base --is-ancestor 9effd097 HEAD
EXIT 0
```

```text
9effd097 feat(w33 G1/S1.7 M1.5): remove adaptive skew classifier surface
15 files changed, 311 insertions(+), 3454 deletions(-)
```

Current code evidence:

```text
crates/xlog-core/src/config.rs
```

`RuntimeConfig` no longer contains `wcoj_cost_model`, `CostModelKind`, or
`resolved_wcoj_cost_model`.

```text
crates/xlog-runtime/src/executor/wcoj_cost_model.rs
```

`build_wcoj_cost_model(_config: &RuntimeConfig)` ignores config/env and returns
`CardinalityAwareCostModel::default()`.

Search confirms the M_INT.5 opt-out surface is absent from current code:

```text
rg -n "XLOG_WCOJ_COST_MODEL|CostModelKind|SkewClassifier|with_wcoj_cost_model|resolved_wcoj_cost_model" \
  crates/xlog-core crates/xlog-runtime crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs
EXIT 1
```

## Verdict

This was the pre-fix blocker. M_INT.5 was not green under the written goal-038
acceptance cell at this point.

The literal command is uncovered, and the acceptance clause
`env skew opt-out preserved` conflicts with the current integrated code after
`9effd097`. Making M_INT.5 green as written would require either reintroducing
the removed skew-classifier/`XLOG_WCOJ_COST_MODEL=skew` surface or amending the
M_INT.5 acceptance cell to a successor post-G1 contract.

Follow-up `docs/evidence/2026-05-14-g38-int-mint5-default-flip.md` restores the
selector API, adds the named cert, and makes M_INT.5 green with the documented
post-G1 caveat.
