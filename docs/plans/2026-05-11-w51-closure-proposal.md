# W5.1 Closure Proposal - Cert Trio

**Date:** 2026-05-11
**Branch:** `feat/w51-cert-trio`
**Base:** `main` at `ffe27f4d`
**Plan:** `docs/plans/2026-05-11-w51-cert-trio-plan.md` at `85d62e84`
**Evidence:** `docs/evidence/2026-05-11-w51-cert-trio/README.md`

---

## Status

W5.1 is implemented as a cert-only branch. It adds exactly three integration test
files for the closure-board W5.1 acceptance line and does not edit the closure
board, push, tag, merge, add benchmarks, or change production runtime surfaces.

Commit count is anchored to the closure-proposal commit that contains this file,
not to any later live branch head. At that commit, W5.1 contains 5 commits on top
of `ffe27f4d`: the plan commit `85d62e84`, three cert commits (`c9da65cb`,
`ae0b4d66`, `69614707`), and this evidence/proposal commit. The review request
resolves the exact closure-proposal commit hash after the commit exists.

---

## Delivered Work

| Board criterion element | Delivered evidence |
|-------------------------|--------------------|
| Three new test files in `xlog-integration/tests/` | `test_w51_same_generation_gpu_cert.rs`, `test_w51_skewed_multiway_gpu_cert.rs`, `test_w51_deep_recursive_wcoj_cert.rs` |
| Row-set parity vs CPU oracle | Same Generation compares GPU `sg_cert(W,Z)` to local `sg_reference`; skewed multiway compares GPU output to `evaluate_rule_typed`; deep-recursive compares GPU `tri` output to a local host fixpoint walker |
| Dispatch counter greater than zero | Strengthened to exact counter equality: 4-cycle `== 1` for Same Generation; triangle `== 1` for skewed multiway; triangle `== 6` for deep-recursive |
| Non-empty parity | Each cert asserts the CPU oracle row set is non-empty before GPU parity comparison |

---

## Verbatim Plan Locks

Verbatim D7 from plan commit `85d62e84`:

| **D7** | **LOCKED: non-empty parity.** | Each cert must assert the CPU oracle row set is non-empty before comparing GPU output. Empty equality is not closure evidence. |

Verbatim Acceptance Grid from plan commit `85d62e84`:

| Cert | Counter assertion | Parity oracle source | Fixture description | Expected row-set size | Deterministic seed/order |
|------|-------------------|----------------------|---------------------|-----------------------|--------------------------|
| Same Generation GPU cert | `wcoj_4cycle_dispatch_count() == 1`; triangle/clique counters `== 0` | `sg_reference` semantics from `crates/xlog-logic/tests/test_hypergraph_certification.rs:200-238`; GPU output is `sg_cert(W,Z)` | Parent pairs `[(1,10),(2,10),(11,1),(12,2),(13,1),(14,12)]`; final SG is reconstructed from base sibling pairs plus one 4-cycle witness over final `sg`, `parent`, `parent_rev`, `all_child_pairs` | 14 SG pairs | Parent rows sorted as listed; `all_child_pairs` lexicographic over child ids `[1,2,11,12,13,14]`; BTreeSet comparison |
| Skewed multiway GPU cert | `wcoj_triangle_dispatch_count() == 1`; 4-cycle/clique counters `== 0` | `evaluate_rule_typed` over the skewed oracle fixture from `crates/xlog-logic/tests/test_hypergraph_certification.rs:439-493` | `big(X,Y)` for `1..=8` excluding diagonal, `small_a={(2,10),(3,20),(4,30),(5,40)}`, `small_b={(1,10),(2,20),(3,30),(4,40)}` | 4 triples | Generate `big` in lexicographic `x,y` order; fixed `small_a`/`small_b` order; sorted tuple comparison |
| Deep-recursive WCOJ cert | `wcoj_triangle_dispatch_count() == 6`; 4-cycle/clique counters `== 0` | Local host fixpoint walker for D5, modeled after the deep frontier oracle style at `crates/xlog-logic/tests/test_hypergraph_certification.rs:507-579`; optional diagnostic gate-off executor run | Linear recursive triangle chain: seed `(1,2)`, `e2` chain through 6, `e3` closing edges from 1 to 3..6 | 4 triples | Fixed chain rows in ascending order; BTreeSet comparison; no random seed |

---

## Cert Metrics

| Cert | Commit | Result | Row-set size | Counter values |
|------|--------|--------|--------------|----------------|
| Same Generation GPU cert | `c9da65cb` | 1 passed, 0 failed | 14 pairs | `wcoj_4cycle_dispatch_count == 1`; `wcoj_triangle_dispatch_count == 0` |
| Skewed multiway GPU cert | `ae0b4d66` | 1 passed, 0 failed | 4 triples | `wcoj_triangle_dispatch_count == 1`; `wcoj_4cycle_dispatch_count == 0`; `wcoj_clique5_dispatch_count == 0`; `wcoj_clique6_dispatch_count == 0` |
| Deep-recursive WCOJ cert | `69614707` | 1 passed, 0 failed | 4 triples | `wcoj_triangle_dispatch_count == 6`; `wcoj_4cycle_dispatch_count == 0`; `wcoj_clique5_dispatch_count == 0`; `wcoj_clique6_dispatch_count == 0` |

The deep-recursive counter matched the risk-register value exactly:
`wcoj_triangle_dispatch_count == 6`. No iteration-2 plan amendment was needed.

---

## Verification

| Gate | Command | Result |
|------|---------|--------|
| Targeted W5.1 certs | `cargo test -p xlog-integration --release --test test_w51_same_generation_gpu_cert --test test_w51_skewed_multiway_gpu_cert --test test_w51_deep_recursive_wcoj_cert -- --nocapture` | exit 0; three test binaries, each 1 passed and 0 failed |
| Formatting | `cargo fmt --check --all` | exit 0 |
| CUDA certification suite | `cargo test -p xlog-cuda-tests --test certification_suite --release` | exit 0; 1 passed, 0 failed |
| Workspace build | `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | exit 0 |
| Canonical workspace test | `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | exit 0 |

F-W43 exception accounting: no exception was consumed because the canonical
workspace test exited 0. The enumerated files
`test_wcoj_layout_fast_path.rs`, `test_wcoj_layout_u32.rs`, and
`test_wcoj_layout_u64.rs` passed in this run, and the sibling
`test_wcoj_layout_sort_*` files also passed.

Bench gate: none for W5.1. This proposal makes no benchmark ratio claim; W5.2
owns benchmark evidence.

---

## Closure-Board Question

Does the closure board accept W5.1 as satisfying the stated certification
criterion?

Board acceptance line from the plan:

`W5.1 | ROADMAP item #17 | OPEN | - | GPU Same Generation cert (currently oracle-only); skewed multiway GPU cert; deep-recursive WCOJ cert. | Three new test files in xlog-integration/tests/; each asserts row-set parity vs. CPU oracle and dispatch counter > 0.`

W5.1 meets the line with exact equality rather than only `> 0` counter checks.

---

## Response Options

| # | Response | Implication |
|---|----------|-------------|
| **1** | **Accept as DONE (Recommended)** | W5.1 delivered the three requested cert files, non-empty CPU parity, exact dispatch counters, and clean verification gates. A later board-update commit may move W5.1 from OPEN to DONE after supervisor approval. |
| **2** | **Reject** | Keep W5.1 OPEN and specify which cert, counter, oracle, fixture, or gate must be corrected in a new implementation commit. |
| **3** | **Defer** | Keep W5.1 OPEN and carry the closure decision forward without changing this branch's delivered cert evidence. |

No Response option is executed by this commit. This commit only files evidence
and asks for supervisor review.
