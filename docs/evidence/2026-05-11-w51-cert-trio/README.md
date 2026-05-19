# W5.1 Cert Trio Evidence

**Date:** 2026-05-11
**Branch:** `feat/w51-cert-trio`
**Plan:** `docs/plans/2026-05-11-w51-cert-trio-plan.md` at `85d62e84`
**Scope:** Certification only. No runtime surfaces, kernels, cost-model changes, board edits, tags, pushes, merges, or benchmark claims.

---

## Summary

W5.1 adds three GPU WCOJ certification tests under `crates/xlog-integration/tests/`.
Each cert compares a GPU-dispatched row set against a deterministic CPU oracle, asserts
the CPU oracle is non-empty, and records exact dispatch-counter evidence.

| Cert | File | Test | Commit | CPU oracle | Row-set size | Dispatch evidence |
|------|------|------|--------|------------|--------------|-------------------|
| Same Generation GPU cert | `crates/xlog-integration/tests/test_w51_same_generation_gpu_cert.rs` | `same_generation_gpu_4cycle_witness_matches_cpu_oracle` | `c9da65cb` | Local `sg_reference` semantics from `crates/xlog-logic/tests/test_hypergraph_certification.rs:200-238` | 14 pairs | `wcoj_4cycle_dispatch_count() == 1`; `wcoj_triangle_dispatch_count() == 0` |
| Skewed multiway GPU cert | `crates/xlog-integration/tests/test_w51_skewed_multiway_gpu_cert.rs` | `skewed_multiway_gpu_triangle_matches_typed_cpu_oracle` | `ae0b4d66` | `evaluate_rule_typed` over the locked skewed fixture from `crates/xlog-logic/tests/test_hypergraph_certification.rs:439-493` | 4 triples | `wcoj_triangle_dispatch_count() == 1`; `wcoj_4cycle_dispatch_count() == 0`; `wcoj_clique5_dispatch_count() == 0`; `wcoj_clique6_dispatch_count() == 0` |
| Deep-recursive WCOJ cert | `crates/xlog-integration/tests/test_w51_deep_recursive_wcoj_cert.rs` | `deep_recursive_triangle_dispatches_exact_count_and_matches_cpu_oracle` | `69614707` | Local host fixpoint walker for the D5 linear-recursive triangle chain | 4 triples | `wcoj_triangle_dispatch_count() == 6`; `wcoj_4cycle_dispatch_count() == 0`; `wcoj_clique5_dispatch_count() == 0`; `wcoj_clique6_dispatch_count() == 0` |

All three certs used TDD red/green execution. The committed cert files are the green
state after each failing RED probe was restored to the intended GPU-dispatched output.

---

## Targeted W5.1 Gate

Command:

```bash
cargo test -p xlog-integration --release \
  --test test_w51_same_generation_gpu_cert \
  --test test_w51_skewed_multiway_gpu_cert \
  --test test_w51_deep_recursive_wcoj_cert \
  -- --nocapture
```

Result: exit 0.

| Test binary | Result | Captured W5.1 metrics |
|-------------|--------|------------------------|
| `test_w51_deep_recursive_wcoj_cert` | 1 passed, 0 failed | `wcoj_triangle_dispatch_count=6`; 4-cycle/clique counters 0; `row_set_size=4` |
| `test_w51_same_generation_gpu_cert` | 1 passed, 0 failed | `wcoj_4cycle_dispatch_count=1`; `wcoj_triangle_dispatch_count=0`; `row_set_size=14` |
| `test_w51_skewed_multiway_gpu_cert` | 1 passed, 0 failed | `wcoj_triangle_dispatch_count=1`; 4-cycle/clique counters 0; `row_set_size=4` |

---

## Repository Verification

| Gate | Command | Result |
|------|---------|--------|
| Formatting | `cargo fmt --check --all` | exit 0 |
| CUDA certification suite | `cargo test -p xlog-cuda-tests --test certification_suite --release` | exit 0; 1 passed, 0 failed |
| Workspace build | `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | exit 0 |
| Canonical workspace test | `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | exit 0 |

F-W43 exception accounting: the canonical workspace test exited 0, so no flake
exception was consumed. The three enumerated exception files
`test_wcoj_layout_fast_path.rs`, `test_wcoj_layout_u32.rs`, and
`test_wcoj_layout_u64.rs` passed in this run, and the non-exempt sibling files
`test_wcoj_layout_sort_roundtrip.rs`, `test_wcoj_layout_sort_u32.rs`, and
`test_wcoj_layout_sort_u64.rs` also passed.

---

## No Benchmark Claim

W5.1 is a certification slice. It proves row-set parity and dispatch-counter
coverage for the three requested GPU WCOJ certs. It does not claim a speedup or
benchmark ratio; W5.2 owns benchmark evidence.

---

## Process Notes

* The closure board is not edited by W5.1 implementation.
* The W5.1 closure proposal is `docs/plans/2026-05-11-w51-closure-proposal.md`.
* Any OPEN to DONE transition remains a separate supervisor decision after review.
