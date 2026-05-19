# W5.1 Cert Trio Plan (iteration 1 canonical)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three GPU WCOJ certification tests for Same Generation, skewed multiway, and deep-recursive WCOJ, each with exact dispatch-counter evidence and row-set parity against a CPU oracle.

**Architecture:** W5.1 is cert-only. It reuses the current executor WCOJ surfaces (triangle, 4-cycle, recursive triangle) and does not add kernels, runtime knobs, cost-model behavior, board edits, or benchmarks. The tests compare GPU-dispatched output to deterministic host/oracle row sets and assert exact dispatch counts.

**Tech Stack:** Rust integration tests in `crates/xlog-integration/tests/`, `xlog-logic` typed oracle helpers, `xlog-runtime::Executor`, `RuntimeConfig`, CUDA-backed `CudaKernelProvider`.

---

## Acceptance Line

From `docs/v065-closure-board.md:104`:

`W5.1 | ROADMAP item #17 | OPEN | - | GPU Same Generation cert (currently oracle-only); skewed multiway GPU cert; deep-recursive WCOJ cert. | Three new test files in xlog-integration/tests/; each asserts row-set parity vs. CPU oracle and dispatch counter > 0.`

Related roadmap line: `ROADMAP.md:1266-1268` says the GPU Same Generation / skewed multiway / deep-recursive WCOJ execution gates are still open because the existing certification is oracle-layer only.

## Paper-Alignment Note

Read `reference_srdatalog_paper.md` at `/home/dev/.claude/projects/-home-dev-projects-xlog/memory/reference_srdatalog_paper.md:19-39` and the W3 audit at `/home/dev/projects/xlog/.worktrees/w3-paper-alignment-audit/docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md:17-23`.

* Same Generation validates P1 at the certification layer by comparing the GPU-reconstructed Same Generation row set against the independent semi-naive CPU Same Generation oracle. The GPU WCOJ work is a 4-cycle witness over the final `sg` relation; the recursive SG fixed point itself remains the CPU/binary side of the comparison in this iteration.
* Skewed multiway validates P2/P5 for a non-recursive triangle-shaped WCOJ workload: deterministic count/materialize GPU WCOJ over flat columnar inputs, with no histogram claim.
* Deep-recursive WCOJ validates P1/P2/P4/P5 most directly: a recursive triangle body dispatches on delta-substituted variants; the fixture uses canonical slot order so the delta relation is the outermost slot in the current dispatcher path.
* P3 is not claimed by W5.1. Histogram-guided launch-time balancing remains W3.3-owned; this plan must not smuggle a histogram or benchmark claim into a certification item.

## Process Rule Compliance

* LOCKED: W5.1 uses worktree `.worktrees/w51-cert-trio` on branch `feat/w51-cert-trio`, created from `main` HEAD `ffe27f4d`.
* LOCKED: This plan iteration is a plan-only commit. No Rust code, no test files, no evidence files, no board edit, no DONE marking.
* LOCKED: No `git push`, no tag, no FF-merge. Supervisor approval is required before any later board-status action.
* LOCKED: No performance claim. W5.1 is not bench-spike-first work because it closes certification gates only; W5.2 owns benchmark evidence.
* LOCKED: Commit subject for this plan iteration is `docs(plan): W5.1 iteration 1 — cert trio (Same Gen + Skewed Multiway + Deep Recursive)`.

## Read-Only Surface

### CPU Oracle Sources

* Oracle stack overview: `crates/xlog-logic/src/hypergraph/mod.rs:39-42` lists certification workloads covering Same Generation, skewed multiway, and deep recursive frontier.
* CPU evaluator contract: `crates/xlog-logic/src/hypergraph/reference.rs:1-5` and `:297-302` define the deterministic WCOJ correctness oracle and fixture-validation boundary.
* Same Generation oracle: `crates/xlog-logic/tests/test_hypergraph_certification.rs:200-238` is the independent nested-loop `sg_reference`; `:240-313` is the current oracle-layer Same Generation cert.
* Skewed multiway oracle fixture: `crates/xlog-logic/tests/test_hypergraph_certification.rs:439-493` defines `big`, `small_a`, `small_b` and the expected 4-row result.
* Deep recursive oracle precedent: `crates/xlog-logic/tests/test_hypergraph_certification.rs:507-579` uses `evaluate_fixpoint_typed` for a deep recursive frontier. W5.1 will use a GPU-WCOJ-shaped linear-recursive triangle and a small host oracle in the integration test because the existing oracle frontier is not a triangle/4-cycle GPU-dispatch shape.

### Dispatch Counter and Runtime Surfaces

* Executor counter fields initialize to zero at `crates/xlog-runtime/src/executor/mod.rs:90-106` and `:239-242`.
* Triangle counter increments only after successful GPU WCOJ output at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:969-1004`; accessor at `:1248-1254`.
* 4-cycle counter increments only after successful GPU WCOJ output at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1434-1471`; accessor at `:1257-1263`.
* Clique counters exist at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1768-1782`, but W5.1 does not use clique counters. They remain W5.2/W3.2 context.
* Recursive dispatch helper calls triangle then 4-cycle and increments through the same counter paths; see `crates/xlog-runtime/src/executor/recursive.rs:16-68` and per-variant rewrite dispatch at `:506-527`.
* Non-recursive dispatch tries triangle, 4-cycle, clique5, clique6 before fallback; see `crates/xlog-runtime/src/executor/recursive.rs:120-210`.

### Cert Templates

* Runtime-backed CUDA fixture, upload, download, and `run_program` template: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:71-143`, `:145-241`, `:243-271`.
* Existing recursive triangle template: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:602-695`.
* Existing recursive 4-cycle template: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:705-777`.
* Existing k-clique cert style for default-dispatch + fallback parity: `crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs:1-17` and `:360-410`.
* 4-cycle output matcher requires 4 output columns in certified layouts; see `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:438-457`. Same Generation therefore needs a 4-column witness rule, then a projection to SG pairs for oracle parity.

### Gate Exception Surface

* W5.1 inherits the F-W43-12/F-W43-15 workspace-test exception: the only exempt workspace-test files are `crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs`, `test_wcoj_layout_u32.rs`, and `test_wcoj_layout_u64.rs`; sibling `test_wcoj_layout_sort_*` files are not exempt. See `docs/plans/2026-05-10-w43-sort-merge-join-plan.md:190-196` and `docs/v065-closure-board.md:98`.

## Direction Table

| ID | Lock | Direction |
|----|------|-----------|
| **D1** | **LOCKED: three files, one cert per file.** | Create exactly three W5.1 integration test files: `crates/xlog-integration/tests/test_w51_same_generation_gpu_cert.rs`, `crates/xlog-integration/tests/test_w51_skewed_multiway_gpu_cert.rs`, and `crates/xlog-integration/tests/test_w51_deep_recursive_wcoj_cert.rs`. No helper crate, no production-code movement, no shared module unless a later implementation review proves duplication is unsafe. |
| **D2** | **LOCKED: exact counter assertions.** | Replace the board's soft `> 0` with exact equality in every cert. Same Generation asserts `wcoj_4cycle_dispatch_count() == 1`. Skewed multiway asserts `wcoj_triangle_dispatch_count() == 1`. Deep-recursive WCOJ asserts `wcoj_triangle_dispatch_count() == 6` on the locked 4-output-row chain fixture. Any empirical mismatch during implementation requires a new plan iteration before changing the expected value. |
| **D3** | **LOCKED: Same Generation cert shape.** | Use the existing `sg_reference` semantics as the CPU oracle, with parent pairs `[(1,10), (2,10), (11,1), (12,2), (13,1), (14,12)]` and expected SG row-set size 14. GPU-side program first computes final `sg` by the standard two-rule Same Generation recursion, then runs one 4-cycle WCOJ witness rule `sg_witness(W,X,Y,Z) :- parent(W,X), sg(X,Y), parent_rev(Y,Z), all_child_pairs(Z,W)` plus projection/base-copy rules into `sg_cert(W,Z)`. `all_child_pairs` is the deterministic ordered Cartesian product over the six child ids; it closes the 4-cycle without constraining the SG semantics. |
| **D4** | **LOCKED: skewed multiway cert shape.** | Port the oracle-layer skewed fixture unchanged in semantics: `result(X,Y,Z) :- big(X,Y), small_a(Y,Z), small_b(X,Z)`, with `big = {(x,y) | x,y in 1..=8, x != y}`, four `small_a` rows, four `small_b` rows, expected row-set size 4, and deterministic lexicographic input order. Force triangle dispatch with `RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true))`. CPU parity source is `evaluate_rule_typed` on the same fixture, not a handwritten expected-only shortcut. |
| **D5** | **LOCKED: deep-recursive WCOJ cert shape.** | Extend the existing linear-recursive triangle template to a deeper deterministic chain: `e1_seed = [(1,2)]`, `e2 = [(2,3),(3,4),(4,5),(5,6)]`, `e3 = [(1,3),(1,4),(1,5),(1,6)]`. Expected `tri` row-set size is 4: `(1,2,3)`, `(1,3,4)`, `(1,4,5)`, `(1,5,6)`. Force triangle dispatch on the GPU run; CPU oracle is a small host fixpoint walker for this exact rule, with a secondary gate-off executor run allowed as diagnostic evidence only. |
| **D6** | **LOCKED: no new dispatch surfaces.** | Do not add `RuntimeConfig` fields, env vars, counters, kernels, provider methods, or promoter shapes. If Same Generation cannot satisfy the cert through the 4-cycle witness pattern, stop and write an iteration-2 plan; do not expand runtime scope inside W5.1 iteration 1. |
| **D7** | **LOCKED: non-empty parity.** | Each cert must assert the CPU oracle row set is non-empty before comparing GPU output. Empty equality is not closure evidence. |
| **D8** | **LOCKED: gate compatibility.** | Final implementation gates include targeted runs for all three new test files, `cargo fmt --check --all`, CUDA certification suite 1/1, and the canonical workspace command with only the three enumerated F-W43 exception files excluded from the success claim. No board edit or DONE marking after implementation; closure proposal asks supervisor to review. |

## Step-by-Step Execution Plan

### Step 1 - Spec intake and plan iteration

- [ ] Confirm the active worktree is `.worktrees/w51-cert-trio` on `feat/w51-cert-trio`.
- [ ] Read `docs/plans/2026-05-11-supervisor-goal-001.md`, `docs/v065-closure-board.md:104`, and this plan.
- [ ] Commit this plan iteration only.

Run:

```bash
git status --short --branch
git log -1 --oneline
```

Expected: branch `feat/w51-cert-trio`, HEAD parent `ffe27f4d` before the plan commit.

### Step 2 - Same Generation GPU cert

- [ ] Write failing test `same_generation_gpu_4cycle_witness_matches_cpu_oracle`.
- [ ] Implement local CPU oracle by copying/adapting `sg_reference` semantics from `crates/xlog-logic/tests/test_hypergraph_certification.rs:200-238`.
- [ ] Build GPU program with final `sg` plus 4-cycle witness `sg_witness` and projected `sg_cert`.
- [ ] Assert CPU oracle size `== 14`, GPU `sg_cert` pairs equal CPU oracle, `wcoj_4cycle_dispatch_count() == 1`, and `wcoj_triangle_dispatch_count() == 0`.

Run:

```bash
cargo test -p xlog-integration --release --test test_w51_same_generation_gpu_cert -- --nocapture
```

Expected after implementation: 1/1 pass.

Commit subject:

```bash
test(w51): add Same Generation GPU 4-cycle witness cert
```

### Step 3 - Skewed multiway GPU cert

- [ ] Write failing test `skewed_multiway_gpu_triangle_matches_typed_cpu_oracle`.
- [ ] Build CPU oracle via `evaluate_rule_typed` using the existing skewed fixture semantics.
- [ ] Execute the same rule with forced triangle dispatch.
- [ ] Assert CPU oracle size `== 4`, GPU result equals CPU result, `wcoj_triangle_dispatch_count() == 1`, and `wcoj_4cycle_dispatch_count() == 0`.

Run:

```bash
cargo test -p xlog-integration --release --test test_w51_skewed_multiway_gpu_cert -- --nocapture
```

Expected after implementation: 1/1 pass.

Commit subject:

```bash
test(w51): add skewed multiway GPU triangle cert
```

### Step 4 - Deep-recursive WCOJ cert

- [ ] Write failing test `deep_recursive_triangle_dispatches_exact_count_and_matches_cpu_oracle`.
- [ ] Implement a host fixpoint oracle for the locked linear-recursive triangle chain in D5.
- [ ] Execute GPU program with forced triangle dispatch.
- [ ] Assert CPU oracle size `== 4`, GPU `tri` row set equals CPU oracle, `wcoj_triangle_dispatch_count() == 6`, and `wcoj_4cycle_dispatch_count() == 0`.

Run:

```bash
cargo test -p xlog-integration --release --test test_w51_deep_recursive_wcoj_cert -- --nocapture
```

Expected after implementation: 1/1 pass.

Commit subject:

```bash
test(w51): add deep-recursive WCOJ exact-counter cert
```

### Step 5 - Aggregate W5.1 targeted cert gate

- [ ] Run all three W5.1 cert files together.

Run:

```bash
cargo test -p xlog-integration --release \
  --test test_w51_same_generation_gpu_cert \
  --test test_w51_skewed_multiway_gpu_cert \
  --test test_w51_deep_recursive_wcoj_cert \
  -- --nocapture
```

Expected after implementation: 3/3 pass.

### Step 6 - Repository verification

- [ ] Run formatting:

```bash
cargo fmt --check --all
```

- [ ] Run CUDA cert suite:

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release
```

- [ ] Run canonical workspace command with F-W43 exception accounting:

```bash
cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests
```

Expected: all non-exempt paths pass. If one of the three enumerated F-W43 files flakes, record exact file/test and rerun sibling `test_wcoj_layout_sort_*` files to prove the exception did not widen.

### Step 7 - Evidence and closure proposal

- [ ] Add evidence README for W5.1 only after the three certs and verification gates have measured values.
- [ ] Write closure proposal at `docs/plans/2026-05-11-w51-closure-proposal.md` with exact counter values, row-set sizes, commands, and the no-benchmark rationale.
- [ ] Do not edit the closure board.

Commit subject:

```bash
docs(w51): add cert evidence and closure proposal
```

### Step 8 - Supervisor approval gate

- [ ] Post `GOAL G1 COMPLETE - REVIEW REQUEST` after the plan-iteration commit in this goal.
- [ ] For implementation goal 002, post closure proposal after Step 7 and wait for supervisor approval before any board-status change.

## Acceptance Grid

| Cert | Counter assertion | Parity oracle source | Fixture description | Expected row-set size | Deterministic seed/order |
|------|-------------------|----------------------|---------------------|-----------------------|--------------------------|
| Same Generation GPU cert | `wcoj_4cycle_dispatch_count() == 1`; triangle/clique counters `== 0` | `sg_reference` semantics from `crates/xlog-logic/tests/test_hypergraph_certification.rs:200-238`; GPU output is `sg_cert(W,Z)` | Parent pairs `[(1,10),(2,10),(11,1),(12,2),(13,1),(14,12)]`; final SG is reconstructed from base sibling pairs plus one 4-cycle witness over final `sg`, `parent`, `parent_rev`, `all_child_pairs` | 14 SG pairs | Parent rows sorted as listed; `all_child_pairs` lexicographic over child ids `[1,2,11,12,13,14]`; BTreeSet comparison |
| Skewed multiway GPU cert | `wcoj_triangle_dispatch_count() == 1`; 4-cycle/clique counters `== 0` | `evaluate_rule_typed` over the skewed oracle fixture from `crates/xlog-logic/tests/test_hypergraph_certification.rs:439-493` | `big(X,Y)` for `1..=8` excluding diagonal, `small_a={(2,10),(3,20),(4,30),(5,40)}`, `small_b={(1,10),(2,20),(3,30),(4,40)}` | 4 triples | Generate `big` in lexicographic `x,y` order; fixed `small_a`/`small_b` order; sorted tuple comparison |
| Deep-recursive WCOJ cert | `wcoj_triangle_dispatch_count() == 6`; 4-cycle/clique counters `== 0` | Local host fixpoint walker for D5, modeled after the deep frontier oracle style at `crates/xlog-logic/tests/test_hypergraph_certification.rs:507-579`; optional diagnostic gate-off executor run | Linear recursive triangle chain: seed `(1,2)`, `e2` chain through 6, `e3` closing edges from 1 to 3..6 | 4 triples | Fixed chain rows in ascending order; BTreeSet comparison; no random seed |

### Global Verification Gates

| Gate | Command | Expected |
|------|---------|----------|
| Targeted W5.1 certs | `cargo test -p xlog-integration --release --test test_w51_same_generation_gpu_cert --test test_w51_skewed_multiway_gpu_cert --test test_w51_deep_recursive_wcoj_cert -- --nocapture` | 3/3 pass after implementation |
| Formatting | `cargo fmt --check --all` | exit 0 |
| CUDA cert suite | `cargo test -p xlog-cuda-tests --test certification_suite --release` | 1/1 pass |
| Workspace | `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | exit 0 outside the three enumerated F-W43 flake files |
| Bench | none for W5.1 | no benchmark ratio claim; W5.2 owns benchmark evidence |

## Source-of-Truth References

* Supervisor goal: `docs/plans/2026-05-11-supervisor-goal-001.md`.
* Closure board: `docs/v065-closure-board.md:104`.
* Roadmap: `ROADMAP.md:1266-1268`.
* Paper memory pointer: `/home/dev/.claude/projects/-home-dev-projects-xlog/memory/reference_srdatalog_paper.md:19-39`.
* W3 paper audit: `/home/dev/projects/xlog/.worktrees/w3-paper-alignment-audit/docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md:17-23`.
* Existing oracle workloads: `crates/xlog-logic/tests/test_hypergraph_certification.rs`.
* Existing runtime cert templates: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`, `crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs`.
* Current dispatcher/counter implementation: `crates/xlog-runtime/src/executor/wcoj_dispatch.rs`, `crates/xlog-runtime/src/executor/recursive.rs`.

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Same Generation witness is too indirect because recursive SG itself is not WCOJ-dispatched. | Supervisor may reject it as insufficient for "GPU Same Generation". | This plan makes the scope explicit: GPU WCOJ certifies the Same Generation row set reconstruction against `sg_reference` without adding unsupported promoter/runtime shapes. If rejected, iteration 2 must choose between new runtime shape support or a different accepted cert reading. |
| Exact deep-recursive counter differs from `== 6` under real execution. | Cert would fail or plan would be stale. | TDD red/green step must measure the counter. Any mismatch requires plan iteration 2 before changing D2/D5 or the grid. |
| CPU oracle helper duplication drifts across files. | Tests become hard to audit. | Keep helpers small and local for iteration 1. Extract only if implementation review finds meaningful duplication after all three tests exist. |
| F-W43 workspace flake appears during final gate. | Could mask W5.1 regression. | Exception remains enumerated only. Re-run sibling sort-layout files if the workspace command flakes, and report exact victim file/test. |
| Fixture output accidentally empty. | Parity check false-passes. | D7 requires explicit non-empty and exact row-count assertions before parity. |

## Plan-Approval Gate

This iteration 1 plan is ready for supervisor review after the plan-only commit lands on `feat/w51-cert-trio`. It authorizes no implementation by itself. Implementation starts only under the next supervisor goal or explicit approval, and any change to D1-D8 requires a new plan-iteration commit before code changes.
