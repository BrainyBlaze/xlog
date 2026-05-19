# v0.8.0 Integration And Regression Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_INT composed-branch validation.

Validated runtime/code head before this evidence commit: `bfa20ef5`.

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_INT.1 formatting | `cargo fmt --check` pass | PASS | exit 0 |
| M080_INT.2 pyxlog | pyxlog targeted and compatibility tests pass | PASS | Python v080/runtime slice `42 passed`; `cargo check -p pyxlog` exit 0; `cargo test -p pyxlog --lib` `7 passed` |
| M080_INT.3 runtime | xlog-runtime delta and recursive tests pass | PASS | delta recompute `2 passed`; fixpoint `6 passed`; recursive stats `10 passed` |
| M080_INT.4 cuda | xlog-cuda touched-surface tests pass | PASS | exact launcher `3 passed`; build script `4 passed`; W32/W64 source audit `12 passed` |
| M080_INT.5 integration | DTS certification pack pass | PASS | cert verify `symbol_coverage=17/17 signature_drift=0`; M37-A surface `4 passed`; WCOJ clique `7 passed`; WCOJ recursive `8 passed`; W32 promoter `15 passed` |
| M080_INT.6 docs | roadmap, architecture docs, and evidence cross-links consistent | PASS | v080 evidence JSON validates; cross-link scan over roadmap/docs/evidence returns expected v080 evidence references |
| M080_INT.7 git hygiene | no unrelated files, no generated bulk artifacts unless documented | PASS | worktree clean before INT evidence; INT commit is docs/evidence only |

## Validation Commands

| Command | Result |
|---------|--------|
| `cargo fmt --check` | exit 0 |
| `/tmp/xlog-v080-cert-venv/bin/python -m pytest -q python/tests/test_v080_dts_cert.py python/tests/test_v080_pyapi_source.py python/tests/test_v080_delta_source.py python/tests/test_v080_bridge_source.py python/tests/test_v080_exact_source.py python/tests/test_ilp_exact_induce.py python/tests/test_gpu_native_forward_backward_returns_tensor.py python/tests/test_coins.py python/tests/test_tensor_source.py` | exit 0; 42 passed |
| `cargo check -p pyxlog` | exit 0 |
| `cargo test -p pyxlog --lib` | exit 0; 7 passed |
| `/tmp/xlog-v080-cert-venv/bin/python scripts/v080_dts_cert.py verify docs/evidence/2026-05-18-v080-cert/pyxlog_api_manifest.json` | exit 0; `PASS symbol_coverage=17/17 signature_drift=0` |
| `cargo test -p xlog-runtime --lib apply_deltas_and_recompute` | exit 0; 2 passed |
| `cargo test -p xlog-runtime --lib fixpoint` | exit 0; 6 passed |
| `cargo test -p xlog-runtime --test test_w23_recursive_stats --features xlog-runtime/recursive-stats-trace` | exit 0; 10 passed |
| `cargo test -p xlog-cuda --lib ilp_exact` | exit 0; 3 passed |
| `cargo test -p xlog-cuda --test build_script_tests` | exit 0; 4 passed |
| `cargo test -p xlog-cuda --test test_w32_kernel_source_audit` | exit 0; 12 passed |
| `cargo test -p xlog-integration --test test_m37a_surface_preservation` | exit 0; 4 passed |
| `cargo test -p xlog-integration --test test_wcoj_clique_dispatch clique` | exit 0; 7 passed |
| `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch` | exit 0; 8 passed |
| `cargo test -p xlog-logic --test test_w32_clique_promoter clique` | exit 0; 15 passed |
| `json.tool` over v080 runtime/profile/cert evidence JSON files | exit 0 |
| `rg` cross-link scan over `ROADMAP.md`, architecture docs, and v080 evidence | exit 0 |
| `git status --short --branch` before INT evidence | clean |
| `git diff --check` | exit 0 |

## Notes

- No full DTS-DLM pilot was launched for this integration gate.
- No push, tag, release-board update, merge, or final v0.8.0 closure claim is
  authorized by this evidence.
