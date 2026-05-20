# v0.8.5 Language Completeness - PROB_AGG Evidence

**Date:** 2026-05-19
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Base evidence:** `docs/evidence/2026-05-19-v085-magic-sets/README.md`
**Scope:** `G085_PROB_AGG` only. Finite probabilistic aggregate outcomes in exact provenance/PIR and MC aggregate execution.

---

## Changed Files

| File | Purpose |
|------|---------|
| `crates/xlog-prob/src/provenance.rs` | Replaces the generic aggregate rejection with finite exact aggregate outcome enumeration into PIR formulas. |
| `crates/xlog-prob/src/aggregates.rs` | Centralizes aggregate operator state for `count`, `sum`, `min`, `max`, and `logsumexp`. |
| `crates/xlog-prob/src/mc/results.rs` | Reuses the shared aggregate state in MC deterministic aggregate execution. |
| `crates/xlog-prob/src/mc/buffers.rs` | Hardens MC schema inference for probabilistic aggregate predicates and aligns count outputs with the runtime `U64` groupby schema. |
| `crates/xlog-prob/tests/test_v085_prob_aggregates.rs` | Adds exact provenance oracle tests, cap diagnostics, committed-example coverage, and host-io exact/MC GPU coverage. |
| `examples/v085-language/prob_aggregates/finite_outcomes.xlog` | Adds a finite probabilistic aggregate example covering count and numeric aggregate outputs. |
| `docs/language-reference.md`, `docs/architecture/language-v085.md`, `ROADMAP.md`, `CHANGELOG.md` | Records shipped `G085_PROB_AGG` support and current exact-domain cap semantics. |

---

## RED Evidence

The `G085_PROB_AGG` test was introduced before implementation:

| Command | Result |
|---------|--------|
| `cargo test -p xlog-prob --test test_v085_prob_aggregates` | exit 101 before implementation; 3 tests failed on `Provenance extraction does not support aggregation` |

The host-io MC path also exposed schema blockers during implementation:

| Command | Result |
|---------|--------|
| `cargo test -p xlog-prob --features host-io --test test_v085_prob_aggregates` | failed before MC schema hardening on `Inconsistent predicate types for edge`, then on a `U32` vs `U64` aggregate count schema mismatch |

---

## Implementation Summary

Exact probabilistic aggregate support now uses finite outcome enumeration:

```text
probabilistic facts / AD outcomes
  -> provenance row formulas
  -> aggregate group potential rows
  -> finite row-presence masks
  -> aggregate output tuple formulas in PIR
  -> exact D4/GPU evaluation path
```

Implemented in this node:

- finite exact outcome formulas for `count`, `sum`, `min`, `max`, and `logsumexp`;
- `query` and `evidence` references to aggregate output tuples in provenance extraction;
- exact-domain cap diagnostics at 16 uncertain contributing rows per group;
- no materialization of empty probabilistic groups as `count(..., 0)` tuples;
- shared aggregate operator state between exact provenance enumeration and MC deterministic aggregate execution;
- MC schema inference that derives body variable types from known probabilistic fact schemas before declaring aggregate rule heads;
- host-io exact GPU and MC GPU aggregate fixture coverage.

Still gated to later nodes:

- compact lifted aggregate evaluation beyond finite exact enumeration;
- explain JSON fields for aggregate planner/lifting metadata;
- recursive aggregate exact support beyond the existing stratification boundary;
- larger exact aggregate domains unless they are accepted by a later lifting node.

---

## Verification Commands

| Command | Result |
|---------|--------|
| `cargo test -p xlog-prob --test test_v085_prob_aggregates` | exit 0; 4 passed |
| `cargo test -p xlog-prob --features host-io --test test_v085_prob_aggregates` | exit 0; 6 passed |
| `cargo test -p xlog-prob --lib` | exit 0; 56 passed |
| `cargo check --workspace` | exit 0 |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |

---

## G085_PROB_AGG Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_PROB_AGG.1 exact support | finite `count`, `sum`, `min`, `max`, `logsumexp` fixtures supported or typed-declined | PASS | `exact_count_aggregate_provenance_matches_finite_oracle`; `exact_numeric_aggregate_provenance_matches_finite_oracles` |
| M085_PROB_AGG.2 provenance blocker | supported aggregate rules no longer hit generic aggregation blocker | PASS | exact provenance tests extract formulas for aggregate output tuples |
| M085_PROB_AGG.3 oracle parity | exact aggregate probabilities match finite oracle within `1e-9` | PASS | formula probabilities match count/numeric finite oracles within `1e-12` |
| M085_PROB_AGG.4 MC support | MC aggregate fixtures run on GPU and match oracle within confidence interval | PASS | `mc_gpu_count_aggregate_query_matches_finite_oracle` under `--features host-io` |
| M085_PROB_AGG.5 evidence/query | at least one evidence fixture and one query fixture reference aggregate output | PASS | count test includes `evidence(out_degree(1, 2), true)` and aggregate queries |
| M085_PROB_AGG.6 cap diagnostics | exact-domain cap exceeded returns typed diagnostic with remediation | PASS | `exact_aggregate_domain_cap_reports_typed_diagnostic` |

---

## Example Artifacts

| Example | Coverage |
|---------|----------|
| `examples/v085-language/prob_aggregates/finite_outcomes.xlog` | finite probabilistic `count`, `sum`, `min`, `max`, and `logsumexp` outputs with aggregate queries |

The example is extracted by `committed_prob_aggregate_example_extracts_provenance`.

---

## Next Sub-Goal

Proceed to `G085_AGG_LIFT`: aggregate lifting for compact finite domains. No push, tag, release-board update, merge, or v0.8.5 closure is authorized by this evidence.
