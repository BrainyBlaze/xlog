# G086_INT Evidence

## GDSP

- Consumer goal: validate the composed v0.8.6 branch as a release-integration
  candidate for DTS-DLM, neutral Mistaber-style `.xlog` workloads, v0.9.0
  substrate consumers, and public pyxlog users.
- Existing subsystem reused: workspace Cargo checks, crate test suites, Python
  source/runtime guards, v0.8.0/v0.8.5/v0.8.6 example validators,
  no-D2H/no-hidden-transfer guards, JSON validation, package metadata
  validation, and git hygiene checks.
- Scope boundary: this evidence does not authorize merge, push, tag, release
  board updates, or closure. It records integration status for the feature
  branch after the persistent-index performance amendment at validation head
  `df2dbc03`; the implementation evidence head before the hold/correction
  amendment was `0e2a5420`.

## GQM Questions

- Q086_INT.1: all feature-node tests pass together after integration.
- Q086_INT.2: v0.8.0 and v0.8.5 compatibility gates remain green.
- Q086_INT.3: transfer guards cover every accepted v0.8.6 hot path.
- Q086_INT.4: performance gates are backed by raw measurements or explicit
  blockers.
- Q086_INT.5: roadmap, changelog, architecture docs, evidence links, and git
  hygiene are consistent with implemented scope.

## Commands

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo check --workspace` | PASS; finished dev profile in 0.17s |
| `cargo test -p xlog-runtime` | PASS; 142 lib tests, 15 integration tests, 2 doc tests passed, 2 doc tests ignored |
| `cargo test -p xlog-cuda kernel_modules` | PASS; 2 passed |
| `cargo test -p xlog-induce` | PASS; 23 passed |
| `cargo test -p xlog-prob` | PASS; package suite passed, including GPU/native no-D2H guards |
| `cargo test -p xlog-logic` | PASS; package suite passed, including v0.8.5 language and optimizer suites |
| `cargo test -p xlog-integration` | PASS; package suite passed, including strict D2H, cross-mode determinism, WCOJ, and DTS widened-frontier suites |
| `PYTHONPATH=target/debug pytest -q python/tests/test_v080_examples_source.py python/tests/test_v085_examples_source.py python/tests/test_v086_delta_coalescing.py python/tests/test_v086_relation_callbacks.py python/tests/test_v086_relation_callbacks_runtime.py python/tests/test_v086_exact_types_source.py python/tests/test_v086_exact_types_runtime.py python/tests/test_v086_chain_smem_profile_source.py python/tests/test_v086_chain_smem_source.py python/tests/test_v086_cse_source.py python/tests/test_v086_adaptive_reoptimization_source.py python/tests/test_v086_persistent_hash_index_source.py python/tests/test_v086_consumers_source.py python/tests/test_v086_pyxlog_persistent_index_runtime.py` | PASS; 44 passed in 86.31s |
| `python scripts/validate_v086_examples.py` | PASS; v0.8.0 examples 5, v0.8.5 examples 10, v0.8.6 examples 5; `consumer_certification_status=PASS`; `feature_coverage_source=behavior_probes` |
| `python -m json.tool` over v0.8.6 evidence and expected JSON files | PASS |
| `python -m py_compile scripts/validate_v086_examples.py python/tests/test_v086_persistent_hash_index_source.py python/tests/test_v086_consumers_source.py` | PASS |
| `python scripts/validate_package_metadata.py` | PASS |
| `git diff --check` | PASS |

## Metric Interpretation

| Metric | Status | Evidence |
|---|---|---|
| M086_INT.1 formatting | PASS | `cargo fmt --check` exit 0 |
| M086_INT.2 workspace | PASS | `cargo check --workspace` exit 0 |
| M086_INT.3 targeted Rust | PASS | runtime, cuda, induce, prob, logic, and integration crates exited 0 |
| M086_INT.4 Python | PASS | v0.8.0/v0.8.5/v0.8.6 source and runtime guards: 44 passed |
| M086_INT.5 examples | PASS | v0.8.6 validator invoked v0.8.0, v0.8.5, and v0.8.6 validators; example execution passed and consumer certification is behavior-probe backed |
| M086_INT.6 transfer guards | PASS | xlog-prob no-D2H guards, integration strict deterministic D2H tests, and v0.8.6 source/runtime guards passed |
| M086_INT.7 performance | PASS | raw feature-node evidence records delta/CSE/chain/persistent-index performance; persistent-index `speedup_ratio=3.206` with zero tracked DTOH/H2D calls |
| M086_INT.8 docs | PASS | JSON, py_compile, metadata validation, roadmap/changelog/architecture evidence links checked |
| M086_INT.9 git hygiene | PASS | `git diff --check` clean; generated artifacts are limited to refreshed consumer summaries plus this INT evidence |

## Open Issues

None recorded.

## Next Gate

Proceed to G086_CLOSE with `MERGE_READY`. Do not merge, push, tag, or update
release boards without coordinator authorization.
